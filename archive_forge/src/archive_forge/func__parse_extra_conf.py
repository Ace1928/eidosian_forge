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
def _parse_extra_conf(extra_conf):
    if extra_conf:

        def as_pair(config):
            key, val = config.split('=')
            return (key, val)
        list_of_key_val = [as_pair(conf) for conf in extra_conf.split(',')]
        return dict(list_of_key_val)
    return None