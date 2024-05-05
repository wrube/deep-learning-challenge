# deep-learning-challenge
Bootcamp Module 12 homework on deep-learning

## Overview of the analysis: 
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. 

The objective of this project is to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

## Data Locations

#### Notebooks
An exploratory data notebook [initial_model.ipynb](initial_model.ipynb) provides the initial data pre-processing and a simple 2-layer deep-neural network. Additional pre-processing techniques and hyper-parameter tuning are recorded in [AlphabetSoupCharity_Optimisation.ipynb](AlphabetSoupCharity_Optimisation.ipynb).

#### Checkpoints
Model weight checkpoints are located in [checkpoints/](./checkpoints/).

#### Models
Initial and final trained models are located in [models/](./models/).

## Data Preprocessing
### Features in the model
From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organisations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organisation, such as:

- EIN and NAME: Identification columns
- APPLICATION_TYPE: Alphabet Soup application type
- AFFILIATION: Affiliated sector of industry
- CLASSIFICATION: Government organisation classification
- USE_CASE: Use case for funding
- ORGANIZATION: Organisation type
- STATUS: Active status
- INCOME_AMT: Income classification
- SPECIAL_CONSIDERATIONS: Special considerations for application
- ASK_AMT: Funding amount requested
- IS_SUCCESSFUL: Was the money used effectively

### Target Variable
The target variable in this dataset is the binary column IS_SUCCESSFUL. The dataset provide has a nearly 50-50 split between successful donations

```
IS_SUCCESSFUL
1    18261
0    16038
Name: count, dtype: int64
```


### What features were removed?
#### Initial model
In the [initial_model.ipynb](initial_model.ipynb), the identification layers EIN and NAME were removed as they practically do not offer any predictive value. 

Additionally due the large number unique values in the APPLICATIION_TYPES and CLASSIFICATIONS columns, these were binned to reduce the number of individual unique values.
- For the APPLICATION_TYPES column, if there was a total count of any value less than 500 this was pooled into 'Other'
- For the CLASSIFICATIONS column, if there was a total count of any value less than 1000 this was pooled into 'Other'


#### Optimising the model
In the optimisation phase of the project the STATUS column was also dropped due to the rare occurrences of status 0.
```
STATUS
1    34294
0        5
Name: count, dtype: int64
```
The APPLICATIION_TYPES and CLASSIFICATIONS columns were again binned, but less stringently to provide some additional detail in the predictions. 
- For the APPLICATION_TYPES column, if there was a total count of any value less than 50 this was pooled into 'Other'
- For the CLASSIFICATIONS column, if there was a total count of any value less than 10 this was pooled into 'Other'

##### The ASK_AMT column
It was also noted that the ASK_AMT column contains values which are heavily skewed right.

## Compiling Training and Evaluating the Model

### Initial Model
The sugge


### Optimising the Model
What steps did you take in your attempts to increase model performance?

How many neurons, layers, and activation functions did you select for your neural network model, and why?

## Results 

Using bulleted lists and images to support your answers, address the following questions:


## Summary: 
Were you able to achieve the target model performance?
Summarise the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
