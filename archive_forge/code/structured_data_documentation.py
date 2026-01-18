import pathlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union
import pandas as pd
import tensorflow as tf
from tensorflow import nest
from autokeras import auto_model
from autokeras import blocks
from autokeras import nodes as input_module
from autokeras.engine import tuner
from autokeras.tuners import task_specific
from autokeras.utils import types
Search for the best model and hyperparameters for the AutoModel.

        # Arguments
            x: String, numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
                Training data x. If the data is from a csv file, it should be a
                string specifying the path of the csv file of the training data.
            y: String, numpy.ndarray, or tensorflow.Dataset. Training data y.
                If the data is from a csv file, it should be a string, which is the
                name of the target column. Otherwise, It can be raw labels, one-hot
                encoded if more than two classes, or binary encoded for binary
                classification.
            epochs: Int. The number of epochs to train each model during the search.
                If unspecified, we would use epochs equal to 1000 and early stopping
                with patience equal to 30.
            callbacks: List of Keras callbacks to apply during training and
                validation.
            validation_split: Float between 0 and 1. Defaults to 0.2.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling. This argument is
                not supported when `x` is a dataset.
            validation_data: Data on which to evaluate the loss and any model metrics
                at the end of each epoch. The model will not be trained on this data.
                `validation_data` will override `validation_split`. The type of the
                validation data should be the same as the training data.
            **kwargs: Any arguments supported by
                [keras.Model.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).
        # Returns
            history: A Keras History object corresponding to the best model.
                Its History.history attribute is a record of training
                loss values and metrics values at successive epochs, as well as
                validation loss values and validation metrics values (if applicable).
        