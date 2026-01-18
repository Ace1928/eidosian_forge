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
def test_split_basic(self):
    self._check_split(split)
    self.disable_leak_check()
    with self.assertRaises(ValueError) as raises:
        njit(split)(np.ones(5), 2)
    self.assertIn('array split does not result in an equal division', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        njit(split)(np.ones(5), [3], axis=-3)
    self.assertIn('np.split: Argument axis out of bounds', str(raises.exception))