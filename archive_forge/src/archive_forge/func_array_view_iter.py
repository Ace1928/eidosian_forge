import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def array_view_iter(arr, idx):
    total = 0
    for i, v in enumerate(arr[idx]):
        total += i * v
    return total