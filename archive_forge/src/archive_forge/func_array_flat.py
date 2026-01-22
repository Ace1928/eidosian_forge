import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def array_flat(arr, out):
    for i, v in enumerate(arr.flat):
        out[i] = v