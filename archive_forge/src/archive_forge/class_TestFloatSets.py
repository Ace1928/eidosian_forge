import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
class TestFloatSets(TestSets):
    """
    Test sets with floating-point keys.
    """

    def _range(self, stop):
        return np.arange(stop, dtype=np.float32) * np.float32(0.1)