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
def append_column(self, name: str, data: Any) -> Block:
    assert name not in self._table.column_names
    if any((isinstance(item, np.ndarray) for item in data)):
        raise NotImplementedError(f"`{self.__class__.__name__}.append_column()` doesn't support array-like data.")
    return self._table.append_column(name, [data])