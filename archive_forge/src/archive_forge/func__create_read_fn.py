import math
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, List, Optional
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.data.datasource.datasource import Datasource, ReadTask
from ray.util.annotations import PublicAPI
def _create_read_fn(self, num_rows: int, offset: int):

    def read_fn() -> Iterable[Block]:
        with _connect(self.connection_factory) as cursor:
            cursor.execute(f'SELECT * FROM ({self.sql}) as T LIMIT {num_rows} OFFSET {offset}')
            block = _cursor_to_block(cursor)
            return [block]
    return read_fn