import sys
import h5py
import numpy as np
from . import core
def _get_default_fillvalue(dtype):
    kind = np.dtype(dtype).kind
    fillvalue = None
    if kind in ['u', 'i', 'f']:
        size = np.dtype(dtype).itemsize
        fillvalue = default_fillvals[f'{kind}{size}']
    return fillvalue