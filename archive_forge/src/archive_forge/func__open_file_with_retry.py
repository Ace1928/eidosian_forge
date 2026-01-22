import io
import pathlib
import posixpath
import warnings
from typing import (
import numpy as np
import ray
from ray._private.utils import _add_creatable_buckets_param_if_s3_uri
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.util import (
from ray.data.block import Block, BlockAccessor
from ray.data.context import DataContext
from ray.data.datasource.block_path_provider import BlockWritePathProvider
from ray.data.datasource.datasource import Datasource, ReadTask, WriteResult
from ray.data.datasource.file_meta_provider import (
from ray.data.datasource.filename_provider import (
from ray.data.datasource.partitioning import (
from ray.data.datasource.path_util import (
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
def _open_file_with_retry(file_path: str, open_file: Callable[[], 'pyarrow.NativeFile']) -> 'pyarrow.NativeFile':
    """Open file with an exponential backoff retry strategy.

    This is to avoid transient task failure with remote storage (such as S3),
    when the remote storage throttles the requests.
    """
    if OPEN_FILE_MAX_ATTEMPTS < 1:
        raise ValueError(f'OPEN_FILE_MAX_ATTEMPTS cannot be negative or 0. Get: {OPEN_FILE_MAX_ATTEMPTS}')
    return call_with_retry(open_file, match=OPEN_FILE_RETRY_ON_ERRORS, description=f'open file {file_path}', max_attempts=OPEN_FILE_MAX_ATTEMPTS, max_backoff_s=OPEN_FILE_RETRY_MAX_BACKOFF_SECONDS)