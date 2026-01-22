import copy
import itertools
import math
import random
import sys
import unittest
import numpy as np
from numba import jit, njit
from numba.core import utils, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.misc.quicksort import make_py_quicksort, make_jit_quicksort
from numba.misc.mergesort import make_jit_mergesort
from numba.misc.timsort import make_py_timsort, make_jit_timsort, MergeRun
def array_factory(self, lst, shape=None):
    array = np.array(lst, dtype=np.float64)
    if shape is None:
        return array.reshape(-1, array.shape[0])
    else:
        return array.reshape(shape)