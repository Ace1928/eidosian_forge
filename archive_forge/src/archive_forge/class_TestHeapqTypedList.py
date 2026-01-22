import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
class TestHeapqTypedList(_TestHeapq, TestCase):
    """Test heapq with typed lists"""
    listimpl = typed.List