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
def _handle_builtin_metrics_by_model_type(self):
    text_metrics = [token_count(), toxicity(), flesch_kincaid_grade_level(), ari_grade_level()]
    builtin_metrics = []
    if self.model_type in (_ModelType.CLASSIFIER, _ModelType.REGRESSOR):
        self._compute_builtin_metrics()
    elif self.model_type == _ModelType.QUESTION_ANSWERING:
        builtin_metrics = [*text_metrics, exact_match()]
    elif self.model_type == _ModelType.TEXT_SUMMARIZATION:
        builtin_metrics = [*text_metrics, rouge1(), rouge2(), rougeL(), rougeLsum()]
    elif self.model_type == _ModelType.TEXT:
        builtin_metrics = text_metrics
    elif self.model_type == _ModelType.RETRIEVER:
        retriever_k = self.evaluator_config.pop('retriever_k', 3)
        builtin_metrics = [precision_at_k(retriever_k), recall_at_k(retriever_k), ndcg_at_k(retriever_k)]
    self.ordered_metrics = [self._metric_to_metric_tuple(-1, metric) for metric in builtin_metrics]