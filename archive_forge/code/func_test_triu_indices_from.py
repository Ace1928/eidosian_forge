import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def test_triu_indices_from(self):
    self._triangular_indices_from_tests_arr(triu_indices_from_arr)
    self._triangular_indices_from_tests_arr_k(triu_indices_from_arr_k)
    self._triangular_indices_from_exceptions(triu_indices_from_arr, False)
    self._triangular_indices_from_exceptions(triu_indices_from_arr_k, True)