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
def _evaluate_sklearn_model_score_if_scorable(self):
    if self.model_loader_module == 'mlflow.sklearn' and self.raw_model is not None:
        try:
            score = self.raw_model.score(self.X.copy_to_avoid_mutation(), self.y, sample_weight=self.sample_weights)
            self.metrics_values.update(_get_aggregate_metrics_values({'score': score}))
        except Exception as e:
            _logger.warning(f'Computing sklearn model score failed: {e!r}. Set logging level to DEBUG to see the full traceback.')
            _logger.debug('', exc_info=True)