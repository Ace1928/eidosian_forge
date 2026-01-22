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
def _get_binary_sum_up_label_pred_prob(positive_class_index, positive_class, y, y_pred, y_probs):
    y = np.array(y)
    y_bin = np.where(y == positive_class, 1, 0)
    y_pred_bin = None
    y_prob_bin = None
    if y_pred is not None:
        y_pred = np.array(y_pred)
        y_pred_bin = np.where(y_pred == positive_class, 1, 0)
    if y_probs is not None:
        y_probs = np.array(y_probs)
        y_prob_bin = y_probs[:, positive_class_index]
    return (y_bin, y_pred_bin, y_prob_bin)