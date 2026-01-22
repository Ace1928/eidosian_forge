import json
import os
import sys
from typing import Any, Dict
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import (
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.file_utils import _copy_file_or_tree
from mlflow.utils.uri import append_to_uri_path
def _validate_onnx_session_options(onnx_session_options):
    """Validates that the specified onnx_session_options dict is valid.

    Args:
        ort_session_options: The onnx_session_options dict to validate.
    """
    import onnxruntime as ort
    if onnx_session_options is not None:
        if not isinstance(onnx_session_options, dict):
            raise TypeError(f'Argument onnx_session_options should be a dict, not {type(onnx_session_options)}')
        for key, value in onnx_session_options.items():
            if key != 'extra_session_config' and (not hasattr(ort.SessionOptions, key)):
                raise ValueError(f'Key {key} in onnx_session_options is not a valid ONNX Runtime session options key')
            elif key == 'extra_session_config' and (not isinstance(value, dict)):
                raise TypeError(f'Value for key {key} in onnx_session_options should be a dict, not {{type(value)}}')
            elif key == 'execution_mode' and value.upper() not in ['PARALLEL', 'SEQUENTIAL']:
                raise ValueError(f"Value for key {key} in onnx_session_options should be 'parallel' or 'sequential', not {value}")
            elif key == 'graph_optimization_level' and value not in [0, 1, 2, 99]:
                raise ValueError(f'Value for key {key} in onnx_session_options should be 0, 1, 2, or 99, not {value}')
            elif key in ['intra_op_num_threads', 'intra_op_num_threads'] and value < 0:
                raise ValueError(f'Value for key {key} in onnx_session_options should be >= 0, not {value}')