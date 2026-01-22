import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
@njit
def appender(lst, n, raise_at):
    for i in range(n):
        if i == raise_at:
            raise IndexError
        lst.append(i)
    return lst