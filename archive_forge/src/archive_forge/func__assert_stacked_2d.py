import os
import numpy
from numpy import linalg
import cupy
import cupy._util
from cupy import _core
import cupyx
def _assert_stacked_2d(*arrays):
    for a in arrays:
        if a.ndim < 2:
            raise linalg.LinAlgError('{}-dimensional array given. Array must be at least two-dimensional'.format(a.ndim))