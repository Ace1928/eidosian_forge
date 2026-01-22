import collections
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, TypeVar, Union
import numpy as np
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.data._internal.block_builder import BlockBuilder
from ray.data._internal.numpy_support import convert_udf_returns_to_numpy, is_array_like
from ray.data._internal.size_estimator import SizeEstimator
from ray.data.block import Block, BlockAccessor
from ray.data.row import TableRow
class TableBlockBuilder(BlockBuilder):

    def __init__(self, block_type):
        self._columns = collections.defaultdict(list)
        self._column_names = None
        self._tables: List[Any] = []
        self._tables_size_cursor = 0
        self._tables_size_bytes = 0
        self._uncompacted_size = SizeEstimator()
        self._num_rows = 0
        self._num_compactions = 0
        self._block_type = block_type

    def add(self, item: Union[dict, TableRow, np.ndarray]) -> None:
        if isinstance(item, TableRow):
            item = item.as_pydict()
        elif isinstance(item, np.ndarray):
            item = {TENSOR_COLUMN_NAME: item}
        if not isinstance(item, collections.abc.Mapping):
            raise ValueError('Returned elements of an TableBlock must be of type `dict`, got {} (type {}).'.format(item, type(item)))
        item_column_names = item.keys()
        if self._column_names is not None:
            if item_column_names != self._column_names:
                raise ValueError(f'Current row has different columns compared to previous rows. Columns of current row: {sorted(item_column_names)}, Columns of previous rows: {sorted(self._column_names)}.')
        else:
            self._column_names = item_column_names
        for key, value in item.items():
            if is_array_like(value) and (not isinstance(value, np.ndarray)):
                value = np.array(value)
            self._columns[key].append(value)
        self._num_rows += 1
        self._compact_if_needed()
        self._uncompacted_size.add(item)

    def add_block(self, block: Any) -> None:
        if not isinstance(block, self._block_type):
            raise TypeError(f'Got a block of type {type(block)}, expected {self._block_type}.If you are mapping a function, ensure it returns an object with the expected type. Block:\n{block}')
        accessor = BlockAccessor.for_block(block)
        self._tables.append(block)
        self._num_rows += accessor.num_rows()

    @staticmethod
    def _table_from_pydict(columns: Dict[str, List[Any]]) -> Block:
        raise NotImplementedError

    @staticmethod
    def _concat_tables(tables: List[Block]) -> Block:
        raise NotImplementedError

    @staticmethod
    def _empty_table() -> Any:
        raise NotImplementedError

    @staticmethod
    def _concat_would_copy() -> bool:
        raise NotImplementedError

    def will_build_yield_copy(self) -> bool:
        if self._columns:
            return True
        return self._concat_would_copy() and len(self._tables) > 1

    def build(self) -> Block:
        columns = {key: convert_udf_returns_to_numpy(col) for key, col in self._columns.items()}
        if columns:
            tables = [self._table_from_pydict(columns)]
        else:
            tables = []
        tables.extend(self._tables)
        if len(tables) > 0:
            return self._concat_tables(tables)
        else:
            return self._empty_table()

    def num_rows(self) -> int:
        return self._num_rows

    def get_estimated_memory_usage(self) -> int:
        if self._num_rows == 0:
            return 0
        for table in self._tables[self._tables_size_cursor:]:
            self._tables_size_bytes += BlockAccessor.for_block(table).size_bytes()
        self._tables_size_cursor = len(self._tables)
        return self._tables_size_bytes + self._uncompacted_size.size_bytes()

    def _compact_if_needed(self) -> None:
        assert self._columns
        if self._uncompacted_size.size_bytes() < MAX_UNCOMPACTED_SIZE_BYTES:
            return
        columns = {key: convert_udf_returns_to_numpy(col) for key, col in self._columns.items()}
        block = self._table_from_pydict(columns)
        self.add_block(block)
        self._uncompacted_size = SizeEstimator()
        self._columns.clear()
        self._num_compactions += 1