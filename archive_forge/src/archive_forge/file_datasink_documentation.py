import posixpath
import warnings
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional
from ray._private.utils import _add_creatable_buckets_param_if_s3_uri
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.util import _is_local_scheme, call_with_retry
from ray.data.block import Block, BlockAccessor
from ray.data.context import DataContext
from ray.data.datasource.block_path_provider import BlockWritePathProvider
from ray.data.datasource.datasink import Datasink
from ray.data.datasource.filename_provider import (
from ray.data.datasource.path_util import _resolve_paths_and_filesystem
from ray.util.annotations import DeveloperAPI
Write a block of data to a file.

        Args:
            block: The block to write.
            file: The file to write the block to.
        