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

        Download an artifact file or directory to a local directory/file if applicable, and
        return a local path for it.
        The caller is responsible for managing the lifecycle of the downloaded artifacts.

        (self.path contains the base path - hdfs:/some/path/run_id/artifacts)

        Args:
            artifact_path: Relative source path to the desired artifacts file or directory.
            dst_path: Absolute path of the local filesystem destination directory to which
                to download the specified artifacts. This directory must already exist. If
                unspecified, the artifacts will be downloaded to a new, uniquely-named
                directory on the local filesystem.

        Returns:
            Absolute path of the local filesystem location containing the downloaded
            artifacts - file/directory.
        