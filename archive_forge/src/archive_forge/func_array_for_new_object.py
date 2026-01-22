from collections.abc import (
import os
import posixpath
import numpy as np
from .._objects import phil, with_phil
from .. import h5d, h5i, h5r, h5p, h5f, h5t, h5s
from .compat import fspath, filename_encode
def array_for_new_object(data, specified_dtype=None):
    """Prepare an array from data used to create a new dataset or attribute"""
    if is_float16_dtype(specified_dtype):
        as_dtype = specified_dtype
    elif not isinstance(data, np.ndarray) and specified_dtype is not None:
        as_dtype = specified_dtype
    else:
        as_dtype = guess_dtype(data)
    data = np.asarray(data, order='C', dtype=as_dtype)
    if as_dtype is not None:
        data = data.view(dtype=as_dtype)
    return data