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
def test_triu_indices(self):
    self._triangular_indices_tests_n(triu_indices_n)
    self._triangular_indices_tests_n_k(triu_indices_n_k)
    self._triangular_indices_tests_n_m(triu_indices_n_m)
    self._triangular_indices_tests_n_k_m(triu_indices_n_k_m)
    self._triangular_indices_exceptions(triu_indices_n)
    self._triangular_indices_exceptions(triu_indices_n_k)
    self._triangular_indices_exceptions(triu_indices_n_m)
    self._triangular_indices_exceptions(triu_indices_n_k_m)