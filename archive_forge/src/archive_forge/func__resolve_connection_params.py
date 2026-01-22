import os
import posixpath
import tempfile
import urllib.parse
from contextlib import contextmanager
import packaging.version
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import mkdir, relative_path_to_artifact_path
def _resolve_connection_params(artifact_uri):
    parsed = urllib.parse.urlparse(artifact_uri)
    return (parsed.scheme, parsed.hostname, parsed.port, parsed.path)