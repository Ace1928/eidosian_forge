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
@property
def _slices(self):
    offset = 0
    for tables in self.blocks:
        length = len(tables[0])
        yield (offset, length)
        offset += length