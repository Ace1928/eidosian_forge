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
def _log_custom_metric_artifact(self, artifact_name, raw_artifact, custom_metric_tuple):
    """
        This function logs and returns a custom metric artifact. Two cases:
            - The provided artifact is a path to a file, the function will make a copy of it with
              a formatted name in a temporary directory and call mlflow.log_artifact.
            - Otherwise: will attempt to save the artifact to an temporary path with an inferred
              type. Then call mlflow.log_artifact.

        Args:
            artifact_name: the name of the artifact
            raw_artifact: the object representing the artifact
            custom_metric_tuple: an instance of the _CustomMetric namedtuple

        Returns:
            EvaluationArtifact
        """
    exception_and_warning_header = f"Custom artifact function '{custom_metric_tuple.name}' at index {custom_metric_tuple.index} in the `custom_artifacts` parameter"
    inferred_from_path, inferred_type, inferred_ext = _infer_artifact_type_and_ext(artifact_name, raw_artifact, custom_metric_tuple)
    artifact_file_local_path = self.temp_dir.path(artifact_name + inferred_ext)
    if pathlib.Path(artifact_file_local_path).exists():
        raise MlflowException(f"{exception_and_warning_header} produced an artifact '{artifact_name}' that cannot be logged because there already exists an artifact with the same name.")
    if inferred_from_path:
        shutil.copy2(raw_artifact, artifact_file_local_path)
    elif inferred_type is JsonEvaluationArtifact:
        with open(artifact_file_local_path, 'w') as f:
            if isinstance(raw_artifact, str):
                f.write(raw_artifact)
            else:
                json.dump(raw_artifact, f, cls=NumpyEncoder)
    elif inferred_type is CsvEvaluationArtifact:
        raw_artifact.to_csv(artifact_file_local_path, index=False)
    elif inferred_type is NumpyEvaluationArtifact:
        np.save(artifact_file_local_path, raw_artifact, allow_pickle=False)
    elif inferred_type is ImageEvaluationArtifact:
        raw_artifact.savefig(artifact_file_local_path)
    else:
        try:
            with open(artifact_file_local_path, 'wb') as f:
                pickle.dump(raw_artifact, f)
            _logger.warning(f"{exception_and_warning_header} produced an artifact '{artifact_name}' with type '{type(raw_artifact)}' that is logged as a pickle artifact.")
        except pickle.PickleError:
            raise MlflowException(f"{exception_and_warning_header} produced an unsupported artifact '{artifact_name}' with type '{type(raw_artifact)}' that cannot be pickled. Supported object types for artifacts are:\n- A string uri representing the file path to the artifact. MLflow  will infer the type of the artifact based on the file extension.\n- A string representation of a JSON object. This will be saved as a .json artifact.\n- Pandas DataFrame. This will be saved as a .csv artifact.- Numpy array. This will be saved as a .npy artifact.- Matplotlib Figure. This will be saved as an .png image artifact.- Other objects will be attempted to be pickled with default protocol.")
    mlflow.log_artifact(artifact_file_local_path)
    artifact = inferred_type(uri=mlflow.get_artifact_uri(artifact_name + inferred_ext))
    artifact._load(artifact_file_local_path)
    return artifact