import math
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, List, Optional
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.data.datasource.datasource import Datasource, ReadTask
from ray.util.annotations import PublicAPI
def _cursor_to_block(cursor) -> Block:
    import pyarrow as pa
    rows = cursor.fetchall()
    columns = [column_description[0] for column_description in cursor.description]
    pydict = {column: [row[i] for row in rows] for i, column in enumerate(columns)}
    return pa.Table.from_pydict(pydict)