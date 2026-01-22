from numbers import Number
import operator
import os
import threading
import contextlib
import numpy as np
from .pypocketfft import good_size
def _fix_shape_1d(x, n, axis):
    if n < 1:
        raise ValueError(f'invalid number of data points ({n}) specified')
    return _fix_shape(x, (n,), (axis,))