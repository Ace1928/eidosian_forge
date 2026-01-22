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
def _iter_artifacts_recursive(self, path):
    dir_content = [file_info for file_info in self.list_artifacts(path) if file_info.path not in ['.', path]]
    if not dir_content:
        yield FileInfo(path=path, is_dir=True, file_size=None)
        return
    for file_info in dir_content:
        if file_info.is_dir:
            yield from self._iter_artifacts_recursive(file_info.path)
        else:
            yield file_info