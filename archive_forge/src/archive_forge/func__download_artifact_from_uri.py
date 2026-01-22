import os
import pathlib
import posixpath
import tempfile
import urllib.parse
import uuid
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.dbfs_artifact_repo import DbfsRestArtifactRepository
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.os import is_windows
from mlflow.utils.uri import add_databricks_profile_info_to_artifact_uri, append_to_uri_path
def _download_artifact_from_uri(artifact_uri, output_path=None):
    """
    Args:
        artifact_uri: The *absolute* URI of the artifact to download.
        output_path: The local filesystem path to which to download the artifact. If unspecified,
            a local output path will be created.
    """
    root_uri, artifact_path = _get_root_uri_and_artifact_path(artifact_uri)
    return get_artifact_repository(artifact_uri=root_uri).download_artifacts(artifact_path=artifact_path, dst_path=output_path)