import logging
import math
import os
import posixpath
from abc import abstractmethod
from collections import namedtuple
from concurrent.futures import as_completed
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils import chunk_list
from mlflow.utils.file_utils import (
from mlflow.utils.uri import is_fuse_or_uc_volumes_uri
@abstractmethod
def _upload_to_cloud(self, cloud_credential_info, src_file_path, artifact_file_path):
    """
        Upload a single file to the cloud.

        Args:
            cloud_credential_info: ArtifactCredentialInfo object with presigned URL for the file.
            src_file_path: Local source file path for the upload.
            artifact_file_path: Path in the artifact repository where the artifact will be logged.

        """
    pass