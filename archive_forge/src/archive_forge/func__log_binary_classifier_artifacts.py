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
def _log_binary_classifier_artifacts(self):
    from mlflow.models.evaluation.lift_curve import plot_lift_curve
    if self.y_probs is not None:

        def plot_roc_curve():
            self.roc_curve.plot_fn(**self.roc_curve.plot_fn_args)
        self._log_image_artifact(plot_roc_curve, 'roc_curve_plot')

        def plot_pr_curve():
            self.pr_curve.plot_fn(**self.pr_curve.plot_fn_args)
        self._log_image_artifact(plot_pr_curve, 'precision_recall_curve_plot')
        self._log_image_artifact(lambda: plot_lift_curve(self.y, self.y_probs, pos_label=self.pos_label), 'lift_curve_plot')