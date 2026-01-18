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
def test_series_get_index(self):
    i = Index(np.int32([2, 4, 3]))
    s = Series(np.float64([1.5, 4.0, 2.5]), i)
    cfunc = jit(nopython=True)(get_index_usecase)
    got = cfunc(s)
    self.assertIsInstance(got, Index)
    self.assertIs(got._data, i._data)