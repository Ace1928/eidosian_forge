from collections import OrderedDict
import numpy as np
from ..ndarray.sparse import CSRNDArray
from ..ndarray.sparse import array as sparse_array
from ..ndarray import NDArray
from ..ndarray import array
def _slice_along_batch_axis(data, s, batch_axis):
    """Apply slice along the batch axis"""
    ret = data.slice_axis(axis=batch_axis, begin=s.start, end=s.stop)
    return ret