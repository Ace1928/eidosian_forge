import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def array_iter(arr):
    total = 0
    for i, v in enumerate(arr):
        total += i * v
    return total