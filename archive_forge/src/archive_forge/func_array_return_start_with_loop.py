import numpy as np
from numba import typeof, njit
from numba.tests.support import MemoryLeakMixin
import unittest
def array_return_start_with_loop(a):
    for i in range(a.size):
        a[i] += 1
    return a