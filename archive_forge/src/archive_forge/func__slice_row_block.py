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
def _slice_row_block(row_block: List[TableBlock], length: int) -> Tuple[List[TableBlock], List[TableBlock]]:
    sliced = [table.slice(0, length) for table in row_block]
    remainder = [table.slice(length, len(row_block[0]) - length) for table in row_block]
    return (sliced, remainder)