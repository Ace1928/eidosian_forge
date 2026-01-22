import numpy as np
from numba import typeof, njit
from numba.tests.support import MemoryLeakMixin
import unittest
def array_return(a, i):
    a[i] = 123
    return a