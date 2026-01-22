import collections
import heapq
from typing import (
import numpy as np
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.data._internal.table_block import TableBlockAccessor, TableBlockBuilder
from ray.data._internal.util import find_partitions
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.row import TableRow
class PandasRow(TableRow):
    """
    Row of a tabular Dataset backed by a Pandas DataFrame block.
    """

    def __getitem__(self, key: Union[str, List[str]]) -> Any:
        from ray.data.extensions import TensorArrayElement

        def get_item(keys: List[str]) -> Any:
            col = self._row[keys]
            if len(col) == 0:
                return None
            items = col.iloc[0]
            if isinstance(items[0], TensorArrayElement):
                return tuple([item.to_numpy() for item in items])
            try:
                return tuple([item.as_py() for item in items])
            except (AttributeError, ValueError):
                return items
        is_single_item = isinstance(key, str)
        keys = [key] if is_single_item else key
        items = get_item(keys)
        if items is None:
            return None
        elif is_single_item:
            return items[0]
        else:
            return items

    def __iter__(self) -> Iterator:
        for k in self._row.columns:
            yield k

    def __len__(self):
        return self._row.shape[1]