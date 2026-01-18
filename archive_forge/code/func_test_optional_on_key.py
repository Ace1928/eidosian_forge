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
def test_optional_on_key(self):
    a = [3, 1, 4, 1, 5, 9]

    @njit
    def foo(x, predicate):
        if predicate:

            def closure_key(z):
                return 1.0 / z
        else:
            closure_key = None
        new_x = x[:]
        new_x.sort(key=closure_key)
        return (sorted(x[:], key=closure_key), new_x)
    with self.assertRaises(errors.TypingError) as raises:
        TF = True
        foo(a[:], TF)
    msg = 'Key must concretely be None or a Numba JIT compiled function'
    self.assertIn(msg, str(raises.exception))