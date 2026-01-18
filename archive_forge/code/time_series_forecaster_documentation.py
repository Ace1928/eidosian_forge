from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union
import pandas as pd
from autokeras import blocks
from autokeras import nodes as input_module
from autokeras.engine import tuner
from autokeras.tasks import structured_data
from autokeras.tuners import greedy
from autokeras.utils import types
Search for the best model and then predict for remaining data points.

        # Arguments
            x: String, numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
                Training and Test data x. If the data is from a csv file, it
                should be a string specifying the path of the csv file of the
                training data.
            y: String, a list of string(s), numpy.ndarray, pandas.DataFrame or
                tensorflow.Dataset. Training data y.
                If the data is from a csv file, it should be a list of string(s)
                specifying the name(s) of the column(s) need to be forecasted.
                If it is multivariate forecasting, y should be a list of more than
                one column names. If it is univariate forecasting, y should be a
                string or a list of one string.
            validation_split: Float between 0 and 1. Defaults to 0.2.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling. This argument is
                not supported when `x` is a dataset.
                The best model found would be fit on the entire dataset including the
                validation data.
            validation_data: Data on which to evaluate the loss and any model metrics
                at the end of each epoch. The model will not be trained on this data.
                `validation_data` will override `validation_split`. The type of the
                validation data should be the same as the training data.
                The best model found would be fit on the training dataset without the
                validation data.
            **kwargs: Any arguments supported by
                [keras.Model.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).
        