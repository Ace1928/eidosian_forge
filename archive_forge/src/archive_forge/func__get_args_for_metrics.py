import copy
import functools
import inspect
import json
import logging
import math
import pathlib
import pickle
import shutil
import tempfile
import time
import traceback
import warnings
from collections import namedtuple
from functools import partial
from typing import Callable, List, NamedTuple, Optional, Tuple, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from sklearn import metrics as sk_metrics
from sklearn.pipeline import Pipeline as sk_Pipeline
import mlflow
from mlflow import MlflowClient
from mlflow.entities.metric import Metric
from mlflow.exceptions import MlflowException
from mlflow.metrics import (
from mlflow.models.evaluation.artifacts import (
from mlflow.models.evaluation.base import (
from mlflow.models.utils import plot_lines
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.pyfunc import _ServedPyFuncModel
from mlflow.sklearn import _SklearnModelWrapper
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import NumpyEncoder
from mlflow.utils.time import get_current_time_millis
def _get_args_for_metrics(self, metric_tuple, eval_df, input_df) -> Tuple[bool, List[Union[str, pd.DataFrame]]]:
    """
        Given a metric_tuple, read the signature of the metric function and get the appropriate
        arguments from the input/output columns, other calculated metrics, and evaluator_config.

        Args:
            metric_tuple: The metric tuple containing a user provided function and its index
                in the ``extra_metrics`` parameter of ``mlflow.evaluate``.
            eval_df: The evaluation dataframe containing the prediction and target columns.
            input_df: The input dataframe containing the features used to make predictions.

        Returns:
            tuple: A tuple of (bool, list) where the bool indicates if the given metric can
            be calculated with the given eval_df, metrics, and input_df.
                - If the user is missing "targets" or "predictions" parameters when needed, or we
                cannot find a column or metric for a parameter to the metric, return
                    (False, list of missing parameters)
                - If all arguments to the metric function were found, return
                    (True, list of arguments).
        """
    eval_df_copy = eval_df.copy()
    parameters = inspect.signature(metric_tuple.function).parameters
    eval_fn_args = []
    params_not_found = []
    if len(parameters) == 2:
        param_0_name, param_1_name = parameters.keys()
    if len(parameters) == 2 and param_0_name != 'predictions' and (param_1_name != 'targets'):
        eval_fn_args.append(eval_df_copy)
        self._update_aggregate_metrics()
        eval_fn_args.append(copy.deepcopy(self.aggregate_metrics))
    else:
        for param_name, param in parameters.items():
            column = self.col_mapping.get(param_name, param_name)
            if column == 'predictions' or column == self.predictions or column == self.dataset.predictions_name:
                eval_fn_args.append(eval_df_copy['prediction'])
            elif column == 'targets' or column == self.dataset.targets_name:
                if 'target' in eval_df_copy:
                    eval_fn_args.append(eval_df_copy['target'])
                elif param.default == inspect.Parameter.empty:
                    params_not_found.append(param_name)
                else:
                    eval_fn_args.append(param.default)
            elif column == 'metrics':
                eval_fn_args.append(copy.deepcopy(self.metrics_values))
            elif not isinstance(column, str):
                eval_fn_args.append(column)
            elif column in input_df.columns:
                eval_fn_args.append(input_df[column])
            elif self.other_output_columns is not None and column in self.other_output_columns.columns:
                self.other_output_columns_for_eval.add(column)
                eval_fn_args.append(self.other_output_columns[column])
            elif column in self.evaluator_config:
                eval_fn_args.append(self.evaluator_config.get(column))
            elif (metric_value := self._get_column_in_metrics_values(column)):
                eval_fn_args.append(metric_value)
            elif column in [metric_tuple.name for metric_tuple in self.ordered_metrics]:
                eval_fn_args.append(None)
            elif param.default == inspect.Parameter.empty:
                params_not_found.append(param_name)
            else:
                eval_fn_args.append(param.default)
    if len(params_not_found) > 0:
        return (False, params_not_found)
    return (True, eval_fn_args)