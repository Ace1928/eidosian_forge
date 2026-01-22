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
def _download_all_datasets_in_parallel(dataset_location, dst_path):
    num_cpus = os.cpu_count() or _NUM_DEFAULT_CPUS
    with ThreadPoolExecutor(max_workers=min(num_cpus * _NUM_MAX_THREADS_PER_CPU, _NUM_MAX_THREADS)) as executor:
        futures = []
        for location in dataset_location:
            future = executor.submit(_DownloadThenConvertDataset._download_one_dataset, dataset_location=location, dst_path=dst_path)
            futures.append(future)
        dest_locations = []
        failed_downloads = []
        for future in as_completed(futures):
            try:
                dest_locations.append(future.result())
            except Exception as e:
                failed_downloads.append(repr(e))
        if len(failed_downloads) > 0:
            raise MlflowException('During downloading of the datasets a number ' + f'of errors have occurred: {failed_downloads}')
        return dest_locations