import logging
import os
import posixpath
from abc import ABC, ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from mlflow.entities.file_info import FileInfo
from mlflow.entities.multipart_upload import CreateMultipartUploadResponse, MultipartUploadPart
from mlflow.environment_variables import MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.utils.annotations import developer_stable
from mlflow.utils.file_utils import ArtifactProgressBar, create_tmp_dir
from mlflow.utils.validation import bad_path_message, path_not_unique
def _create_download_destination(self, src_artifact_path, dst_local_dir_path=None):
    """
        Creates a local filesystem location to be used as a destination for downloading the artifact
        specified by `src_artifact_path`. The destination location is a subdirectory of the
        specified `dst_local_dir_path`, which is determined according to the structure of
        `src_artifact_path`. For example, if `src_artifact_path` is `dir1/file1.txt`, then the
        resulting destination path is `<dst_local_dir_path>/dir1/file1.txt`. Local directories are
        created for the resulting destination location if they do not exist.

        Args:
            src_artifact_path: A relative, POSIX-style path referring to an artifact stored
                within the repository's artifact root location. `src_artifact_path` should be
                specified relative to the repository's artifact root location.
            dst_local_dir_path: The absolute path to a local filesystem directory in which the
                local destination path will be contained. The local destination path may be
                contained in a subdirectory of `dst_root_dir` if `src_artifact_path` contains
                subdirectories.

        Returns:
            The absolute path to a local filesystem location to be used as a destination
            for downloading the artifact specified by `src_artifact_path`.
        """
    src_artifact_path = src_artifact_path.rstrip('/')
    dirpath = posixpath.dirname(src_artifact_path)
    local_dir_path = os.path.join(dst_local_dir_path, dirpath)
    local_file_path = os.path.join(dst_local_dir_path, src_artifact_path)
    if not os.path.exists(local_dir_path):
        os.makedirs(local_dir_path, exist_ok=True)
    return local_file_path