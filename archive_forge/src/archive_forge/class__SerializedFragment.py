import logging
from typing import (
import numpy as np
import ray
import ray.cloudpickle as cloudpickle
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.util import _check_pyarrow_version, _is_local_scheme
from ray.data.block import Block
from ray.data.context import DataContext
from ray.data.datasource import Datasource
from ray.data.datasource._default_metadata_providers import (
from ray.data.datasource.datasource import ReadTask
from ray.data.datasource.file_meta_provider import (
from ray.data.datasource.partitioning import PathPartitionFilter
from ray.data.datasource.path_util import (
from ray.util.annotations import PublicAPI
class _SerializedFragment:

    def __init__(self, frag: 'ParquetFileFragment'):
        self._data = cloudpickle.dumps((frag.format, frag.path, frag.filesystem, frag.partition_expression))

    def deserialize(self) -> 'ParquetFileFragment':
        import pyarrow.fs
        file_format, path, filesystem, partition_expression = cloudpickle.loads(self._data)
        return file_format.make_fragment(path, filesystem, partition_expression)