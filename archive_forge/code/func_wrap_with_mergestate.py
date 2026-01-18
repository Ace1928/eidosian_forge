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
def wrap_with_mergestate(self, timsort, func, _cache={}):
    """
        Wrap *func* into another compiled function inserting a runtime-created
        mergestate as the first function argument.
        """
    key = (timsort, func)
    if key in _cache:
        return _cache[key]
    merge_init = timsort.merge_init

    @timsort.compile
    def wrapper(keys, values, *args):
        ms = merge_init(keys)
        res = func(ms, keys, values, *args)
        return res
    _cache[key] = wrapper
    return wrapper