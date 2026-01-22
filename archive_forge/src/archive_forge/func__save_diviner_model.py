import logging
import os
import pathlib
import shutil
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import yaml
import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import MLFLOW_DFS_TMP
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.uri import dbfs_hdfs_uri_to_fuse_path, generate_tmp_dfs_path
def _save_diviner_model(diviner_model, path, **kwargs) -> bool:
    """
    Saves a Diviner model to the specified path. If the model was fit by using a Pandas DataFrame
    for the training data submitted to `fit`, directly save the Diviner model object.
    If the Diviner model was fit by using a Spark DataFrame, save the model components separately.
    The metadata and ancillary files to write (JSON and Pandas DataFrames) are written directly
    to a fuse mount location, which the Spark DataFrame that contains the individual serialized
    Diviner model objects is written by using the 'dbfs:' scheme path that Spark recognizes.
    """
    save_path = str(path.joinpath(_MODEL_BINARY_FILE_NAME))
    if getattr(diviner_model, '_fit_with_spark', False):
        if not os.path.isabs(path):
            raise MlflowException(f"The save path provided must be a relative path. The path submitted, '{path}' is an absolute path.")
        tmp_path = generate_tmp_dfs_path(kwargs.get('dfs_tmpdir', MLFLOW_DFS_TMP.get()))
        diviner_model._save_model_df_to_path(tmp_path, **kwargs)
        diviner_data_path = os.path.abspath(save_path)
        tmp_fuse_path = dbfs_hdfs_uri_to_fuse_path(tmp_path)
        shutil.move(src=tmp_fuse_path, dst=diviner_data_path)
        diviner_model._save_model_metadata_components_to_path(path=diviner_data_path)
        return True
    diviner_model.save(save_path)
    return False