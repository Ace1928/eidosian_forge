import functools
from typing import Callable, Final, Optional
import numpy
import cupy
@functools.lru_cache
def _min_value_of(dtype):
    if dtype.kind in 'biu':
        return dtype.type(numpy.iinfo(dtype).min)
    elif dtype.kind in 'f':
        return dtype.type(-numpy.inf)
    else:
        raise RuntimeError(f'Unsupported type: {dtype}')