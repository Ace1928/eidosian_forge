import sys
import operator
import numpy as np
from math import prod
import scipy.sparse as sp
from scipy._lib._util import np_long, np_ulong
def upcast_scalar(dtype, scalar):
    """Determine data type for binary operation between an array of
    type `dtype` and a scalar.
    """
    return (np.array([0], dtype=dtype) * scalar).dtype