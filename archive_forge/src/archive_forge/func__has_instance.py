from collections import OrderedDict
import numpy as np
from ..ndarray.sparse import CSRNDArray
from ..ndarray.sparse import array as sparse_array
from ..ndarray import NDArray
from ..ndarray import array
def _has_instance(data, dtype):
    """Return True if ``data`` has instance of ``dtype``.
    This function is called after _init_data.
    ``data`` is a list of (str, NDArray)"""
    for item in data:
        _, arr = item
        if isinstance(arr, dtype):
            return True
    return False