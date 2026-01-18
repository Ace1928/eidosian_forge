import collections
import heapq
import random
from typing import (
import numpy as np
from ray._private.utils import _get_pyarrow_version
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.data._internal.arrow_ops import transform_polars, transform_pyarrow
from ray.data._internal.numpy_support import (
from ray.data._internal.table_block import TableBlockAccessor, TableBlockBuilder
from ray.data._internal.util import _truncated_repr, find_partitions
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.row import TableRow
def to_pandas(self) -> 'pandas.DataFrame':
    from ray.air.util.data_batch_conversion import _cast_tensor_columns_to_ndarrays
    df = self._table.to_pandas()
    ctx = DataContext.get_current()
    if ctx.enable_tensor_extension_casting:
        df = _cast_tensor_columns_to_ndarrays(df)
    return df