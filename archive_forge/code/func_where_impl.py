import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
def where_impl(cond, x, y):
    """
                Generic implementation for other arrays
                """
    shape = cond.shape
    if x.shape != shape or y.shape != shape:
        raise ValueError('all inputs should have the same shape')
    res = np.empty_like(x)
    for idx, c in np.ndenumerate(cond):
        res[idx] = x[idx] if c else y[idx]
    return res