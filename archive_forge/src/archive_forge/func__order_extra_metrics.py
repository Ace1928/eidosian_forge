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
def _order_extra_metrics(self, eval_df):
    remaining_metrics = self.extra_metrics
    input_df = self.X.copy_to_avoid_mutation()
    while len(remaining_metrics) > 0:
        pending_metrics = []
        failed_results = []
        did_append_metric = False
        for metric_tuple in remaining_metrics:
            can_calculate, eval_fn_args = self._get_args_for_metrics(metric_tuple, eval_df, input_df)
            if can_calculate:
                self.ordered_metrics.append(metric_tuple)
                did_append_metric = True
            else:
                pending_metrics.append(metric_tuple)
                failed_results.append((metric_tuple.name, eval_fn_args))
        if not did_append_metric:
            self._raise_exception_for_malformed_metrics(failed_results, eval_df)
        remaining_metrics = pending_metrics