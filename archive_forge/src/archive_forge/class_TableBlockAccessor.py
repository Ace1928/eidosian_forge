import collections
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, TypeVar, Union
import numpy as np
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.data._internal.block_builder import BlockBuilder
from ray.data._internal.numpy_support import convert_udf_returns_to_numpy, is_array_like
from ray.data._internal.size_estimator import SizeEstimator
from ray.data.block import Block, BlockAccessor
from ray.data.row import TableRow
class TableBlockAccessor(BlockAccessor):
    ROW_TYPE: TableRow = TableRow

    def __init__(self, table: Any):
        self._table = table

    def _get_row(self, index: int, copy: bool=False) -> Union[TableRow, np.ndarray]:
        base_row = self.slice(index, index + 1, copy=copy)
        row = self.ROW_TYPE(base_row)
        return row

    @staticmethod
    def _build_tensor_row(row: TableRow) -> np.ndarray:
        raise NotImplementedError

    def to_default(self) -> Block:
        default = self.to_pandas()
        return default

    def column_names(self) -> List[str]:
        raise NotImplementedError

    def append_column(self, name: str, data: Any) -> Block:
        raise NotImplementedError

    def to_block(self) -> Block:
        return self._table

    def iter_rows(self, public_row_format: bool) -> Iterator[Union[Mapping, np.ndarray]]:
        outer = self

        class Iter:

            def __init__(self):
                self._cur = -1

            def __iter__(self):
                return self

            def __next__(self):
                self._cur += 1
                if self._cur < outer.num_rows():
                    row = outer._get_row(self._cur)
                    if public_row_format and isinstance(row, TableRow):
                        return row.as_pydict()
                    else:
                        return row
                raise StopIteration
        return Iter()

    def _zip(self, acc: BlockAccessor) -> 'Block':
        raise NotImplementedError

    def zip(self, other: 'Block') -> 'Block':
        acc = BlockAccessor.for_block(other)
        if not isinstance(acc, type(self)):
            raise ValueError('Cannot zip {} with block of type {}'.format(type(self), type(other)))
        if acc.num_rows() != self.num_rows():
            raise ValueError('Cannot zip self (length {}) with block of length {}'.format(self.num_rows(), acc.num_rows()))
        return self._zip(acc)

    @staticmethod
    def _empty_table() -> Any:
        raise NotImplementedError

    def _sample(self, n_samples: int, sort_key: 'SortKey') -> Any:
        raise NotImplementedError

    def sample(self, n_samples: int, sort_key: 'SortKey') -> Any:
        if sort_key is None or callable(sort_key):
            raise NotImplementedError(f'Table sort key must be a column name, was: {sort_key}')
        if self.num_rows() == 0:
            return self._empty_table()
        k = min(n_samples, self.num_rows())
        return self._sample(k, sort_key)