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
def _upload_artifact_to_uri(local_path, artifact_uri):
    """Uploads a local artifact (file) to a specified URI.

    Args:
        local_path: The local path of the file to upload.
        artifact_uri: The *absolute* URI of the path to upload the artifact to.

    """
    root_uri, artifact_path = _get_root_uri_and_artifact_path(artifact_uri)
    get_artifact_repository(artifact_uri=root_uri).log_artifact(local_path, artifact_path)