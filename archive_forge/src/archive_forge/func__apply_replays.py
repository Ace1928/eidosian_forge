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
def _apply_replays(table: pa.Table, replays: Optional[List[Replay]]=None) -> pa.Table:
    if replays is not None:
        for name, args, kwargs in replays:
            if name == 'cast':
                table = table_cast(table, *args, **kwargs)
            elif name == 'flatten':
                table = table_flatten(table, *args, **kwargs)
            else:
                table = getattr(table, name)(*args, **kwargs)
    return table