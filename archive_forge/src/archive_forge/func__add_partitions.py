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
def _add_partitions(data: Union['pyarrow.Table', 'pd.DataFrame'], partitions: Dict[str, Any]) -> Union['pyarrow.Table', 'pd.DataFrame']:
    import pandas as pd
    import pyarrow as pa
    assert isinstance(data, (pa.Table, pd.DataFrame))
    if isinstance(data, pa.Table):
        return _add_partitions_to_table(data, partitions)
    if isinstance(data, pd.DataFrame):
        return _add_partitions_to_dataframe(data, partitions)