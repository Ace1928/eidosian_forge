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
def _extend_blocks(result: List[List[TableBlock]], blocks: List[List[TableBlock]], axis: int=0) -> List[List[TableBlock]]:
    if axis == 0:
        result.extend(blocks)
    elif axis == 1:
        result, blocks = _split_both_like(result, blocks)
        for i, row_block in enumerate(blocks):
            result[i].extend(row_block)
    return result