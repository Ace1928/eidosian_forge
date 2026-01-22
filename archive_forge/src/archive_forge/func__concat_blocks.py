import copy
import os
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from . import config
from .utils.logging import get_logger
@staticmethod
def _concat_blocks(blocks: List[Union[TableBlock, pa.Table]], axis: int=0) -> pa.Table:
    pa_tables = [table.table if hasattr(table, 'table') else table for table in blocks]
    if axis == 0:
        if config.PYARROW_VERSION.major < 14:
            return pa.concat_tables(pa_tables, promote=True)
        else:
            return pa.concat_tables(pa_tables, promote_options='default')
    elif axis == 1:
        for i, table in enumerate(pa_tables):
            if i == 0:
                pa_table = table
            else:
                for name, col in zip(table.column_names, table.columns):
                    pa_table = pa_table.append_column(name, col)
        return pa_table
    else:
        raise ValueError("'axis' must be either 0 or 1")