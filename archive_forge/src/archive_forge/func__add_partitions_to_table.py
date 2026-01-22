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
def _add_partitions_to_table(table: 'pyarrow.Table', partitions: Dict[str, Any]) -> 'pyarrow.Table':
    import pyarrow as pa
    import pyarrow.compute as pc
    column_names = set(table.column_names)
    for field, value in partitions.items():
        column = pa.array([value] * len(table))
        if field in column_names:
            column_type = table.schema.field(field).type
            column = column.cast(column_type)
            values_are_equal = pc.all(pc.equal(column, table[field]))
            values_are_equal = values_are_equal.as_py()
            if not values_are_equal:
                raise ValueError(f"Partition column {field} exists in table data, but partition value '{value}' is different from in-data values: {table[field].unique().to_pylist()}.")
            i = table.schema.get_field_index(field)
            table = table.set_column(i, field, column)
        else:
            table = table.append_column(field, column)
    return table