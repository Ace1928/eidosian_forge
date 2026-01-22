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
class StructuredDataClassifier(SupervisedStructuredDataPipeline):
    """AutoKeras structured data classification class.

    # Arguments
        column_names: A list of strings specifying the names of the columns. The
            length of the list should be equal to the number of columns of the data
            excluding the target column. Defaults to None. If None, it will obtained
            from the header of the csv file or the pandas.DataFrame.
        column_types: Dict. The keys are the column names. The values should either
            be 'numerical' or 'categorical', indicating the type of that column.
            Defaults to None. If not None, the column_names need to be specified.
            If None, it will be inferred from the data.
        num_classes: Int. Defaults to None. If None, it will be inferred from the
            data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to use 'binary_crossentropy' or
            'categorical_crossentropy' based on the number of classes.
        metrics: A list of Keras metrics. Defaults to use 'accuracy'.
        project_name: String. The name of the AutoModel. Defaults to
            'structured_data_classifier'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        objective: String. Name of model metric to minimize
            or maximize. Defaults to 'val_accuracy'.
        tuner: String or subclass of AutoTuner. If string, it should be one of
            'greedy', 'bayesian', 'hyperband' or 'random'. It can also be a subclass
            of AutoTuner. If left unspecified, it uses a task specific tuner, which
            first evaluates the most commonly used models for the task before
            exploring other models.
        overwrite: Boolean. Defaults to `False`. If `False`, reloads an existing
            project of the same name if one is found. Otherwise, overwrites the
            project.
        seed: Int. Random seed.
        max_model_size: Int. Maximum number of scalars in the parameters of a
            model. Models larger than this are rejected.
        **kwargs: Any arguments supported by AutoModel.
    """

    def __init__(self, column_names: Optional[List[str]]=None, column_types: Optional[Dict]=None, num_classes: Optional[int]=None, multi_label: bool=False, loss: Optional[types.LossType]=None, metrics: Optional[types.MetricsType]=None, project_name: str='structured_data_classifier', max_trials: int=100, directory: Optional[Union[str, pathlib.Path]]=None, objective: str='val_accuracy', tuner: Union[str, Type[tuner.AutoTuner]]=None, overwrite: bool=False, seed: Optional[int]=None, max_model_size: Optional[int]=None, **kwargs):
        if tuner is None:
            tuner = task_specific.StructuredDataClassifierTuner
        super().__init__(outputs=blocks.ClassificationHead(num_classes=num_classes, multi_label=multi_label, loss=loss, metrics=metrics), column_names=column_names, column_types=column_types, max_trials=max_trials, directory=directory, project_name=project_name, objective=objective, tuner=tuner, overwrite=overwrite, seed=seed, max_model_size=max_model_size, **kwargs)

    def fit(self, x=None, y=None, epochs=None, callbacks=None, validation_split=0.2, validation_data=None, **kwargs):
        """Search for the best model and hyperparameters for the AutoModel.

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
        """
        history = super().fit(x=x, y=y, epochs=epochs, callbacks=callbacks, validation_split=validation_split, validation_data=validation_data, **kwargs)
        return history