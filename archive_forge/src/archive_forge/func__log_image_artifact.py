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
def _log_image_artifact(self, do_plot, artifact_name):
    from matplotlib import pyplot
    prefix = self.evaluator_config.get('metric_prefix', '')
    artifact_file_name = f'{prefix}{artifact_name}.png'
    artifact_file_local_path = self.temp_dir.path(artifact_file_name)
    try:
        pyplot.clf()
        do_plot()
        pyplot.savefig(artifact_file_local_path, bbox_inches='tight')
    finally:
        pyplot.close(pyplot.gcf())
    mlflow.log_artifact(artifact_file_local_path)
    artifact = ImageEvaluationArtifact(uri=mlflow.get_artifact_uri(artifact_file_name))
    artifact._load(artifact_file_local_path)
    self.artifacts[artifact_name] = artifact