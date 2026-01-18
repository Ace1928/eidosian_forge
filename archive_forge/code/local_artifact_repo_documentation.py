import os
import shutil
from mlflow.store.artifact.artifact_repo import ArtifactRepository, verify_artifact_path
from mlflow.utils.file_utils import (
from mlflow.utils.uri import validate_path_is_safe

        Artifacts tracked by ``LocalArtifactRepository`` already exist on the local filesystem.
        If ``dst_path`` is ``None``, the absolute filesystem path of the specified artifact is
        returned. If ``dst_path`` is not ``None``, the local artifact is copied to ``dst_path``.

        Args:
            artifact_path: Relative source path to the desired artifacts.
            dst_path: Absolute path of the local filesystem destination directory to which to
                download the specified artifacts. This directory must already exist. If
                unspecified, the absolute path of the local artifact will be returned.

        Returns:
            Absolute path of the local filesystem location containing the desired artifacts.
        