import importlib
import logging
import os
import pathlib
import posixpath
import sys
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repo import (
from mlflow.utils._spark_utils import (
from mlflow.utils.file_utils import (
@staticmethod
def _download_one_dataset(dataset_location: str, dst_path: str):
    parsed_location_uri = urlparse(dataset_location)
    if parsed_location_uri.scheme in ['http', 'https']:
        dst_file_name = posixpath.basename(parsed_location_uri.path)
        dst_file_path = os.path.join(dst_path, dst_file_name)
        download_file_using_http_uri(http_uri=dataset_location, download_path=dst_file_path, chunk_size=_DownloadThenConvertDataset._FILE_DOWNLOAD_CHUNK_SIZE_BYTES)
        return dst_file_path
    else:
        return download_artifacts(artifact_uri=dataset_location, dst_path=dst_path)