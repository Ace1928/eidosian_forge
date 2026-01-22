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
class StructuredDataRegressor(SupervisedStructuredDataPipeline):
    """AutoKeras structured data regression class.

    # Arguments
        column_names: A list of strings specifying the names of the columns. The
            length of the list should be equal to the number of columns of the data
            excluding the target column. Defaults to None. If None, it will obtained
            from the header of the csv file or the pandas.DataFrame.
        column_types: Dict. The keys are the column names. The values should either
            be 'numerical' or 'categorical', indicating the type of that column.
            Defaults to None. If not None, the column_names need to be specified.
            If None, it will be inferred from the data.
        output_dim: Int. The number of output dimensions. Defaults to None.
            If None, it will be inferred from the data.
        loss: A Keras loss function. Defaults to use 'mean_squared_error'.
        metrics: A list of Keras metrics. Defaults to use 'mean_squared_error'.
        project_name: String. The name of the AutoModel. Defaults to
            'structured_data_regressor'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        objective: String. Name of model metric to minimize
            or maximize, e.g. 'val_accuracy'. Defaults to 'val_loss'.
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

    def __init__(self, column_names: Optional[List[str]]=None, column_types: Optional[Dict[str, str]]=None, output_dim: Optional[int]=None, loss: types.LossType='mean_squared_error', metrics: Optional[types.MetricsType]=None, project_name: str='structured_data_regressor', max_trials: int=100, directory: Union[str, pathlib.Path, None]=None, objective: str='val_loss', tuner: Union[str, Type[tuner.AutoTuner]]=None, overwrite: bool=False, seed: Optional[int]=None, max_model_size: Optional[int]=None, **kwargs):
        if tuner is None:
            tuner = task_specific.StructuredDataRegressorTuner
        super().__init__(outputs=blocks.RegressionHead(output_dim=output_dim, loss=loss, metrics=metrics), column_names=column_names, column_types=column_types, max_trials=max_trials, directory=directory, project_name=project_name, objective=objective, tuner=tuner, overwrite=overwrite, seed=seed, max_model_size=max_model_size, **kwargs)