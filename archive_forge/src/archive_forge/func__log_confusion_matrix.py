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
def _log_confusion_matrix(self):
    """
        Helper method for logging confusion matrix
        """
    confusion_matrix = sk_metrics.confusion_matrix(self.y, self.y_pred, labels=self.label_list, normalize='true', sample_weight=self.sample_weights)

    def plot_confusion_matrix():
        import matplotlib
        import matplotlib.pyplot as plt
        with matplotlib.rc_context({'font.size': min(8, math.ceil(50.0 / self.num_classes)), 'axes.labelsize': 8}):
            _, ax = plt.subplots(1, 1, figsize=(6.0, 4.0), dpi=175)
            disp = sk_metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=self.label_list).plot(cmap='Blues', ax=ax)
            disp.ax_.set_title('Normalized confusion matrix')
    if hasattr(sk_metrics, 'ConfusionMatrixDisplay'):
        self._log_image_artifact(plot_confusion_matrix, 'confusion_matrix')
    return