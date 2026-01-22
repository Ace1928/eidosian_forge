import logging
import math
import pickle
import warnings
import os
import numpy
from ..base import py_str
from ..ndarray import (NDArray, zeros, clip, sqrt, cast, maximum, abs as NDabs, array, multiply,
from ..ndarray import (sgd_update, sgd_mom_update, adam_update, rmsprop_update, rmspropalex_update,
from ..ndarray.contrib import (multi_lamb_update, multi_mp_lamb_update)
from ..ndarray import sparse
from ..random import normal
from ..util import is_np_array
def _as_classic(a, allow_np):
    from ..numpy import ndarray as np_ndarray
    if isinstance(a, (tuple, list)):
        if any((isinstance(x, np_ndarray) for x in a)):
            if allow_np:
                return [x.as_nd_ndarray() for x in a]
            else:
                raise ValueError('Converting np.ndarray to mx.nd.NDArray is not allowed')
    elif isinstance(a, np_ndarray):
        if allow_np:
            return a.as_nd_ndarray()
        else:
            raise ValueError('Converting np.ndarray to mx.nd.NDArray is not allowed')
    return a