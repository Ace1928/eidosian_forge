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
def _raise_exception_for_malformed_metrics(self, malformed_results, eval_df):
    output_columns = [] if self.other_output_columns is None else list(self.other_output_columns.columns)
    if self.predictions:
        output_columns.append(self.predictions)
    elif self.dataset.predictions_name:
        output_columns.append(self.dataset.predictions_name)
    else:
        output_columns.append('predictions')
    input_columns = list(self.X.copy_to_avoid_mutation().columns)
    if 'target' in eval_df:
        if self.dataset.targets_name:
            input_columns.append(self.dataset.targets_name)
        else:
            input_columns.append('targets')
    error_message = self._construct_error_message_for_malformed_metrics(malformed_results, input_columns, output_columns)
    raise MlflowException(error_message, error_code=INVALID_PARAMETER_VALUE)