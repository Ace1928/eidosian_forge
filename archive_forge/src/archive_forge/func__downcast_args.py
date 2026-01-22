from __future__ import annotations
import operator
import numpy as np
import pytest
import dask.array as da
from dask.array import Array
from dask.array.chunk_types import is_valid_array_chunk, is_valid_chunk_type
from dask.array.utils import assert_eq
def _downcast_args(self, args):
    for arg in args:
        if isinstance(arg, type(self)):
            yield arg.arr
        elif isinstance(arg, (tuple, list)):
            yield type(arg)(self._downcast_args(arg))
        else:
            yield arg