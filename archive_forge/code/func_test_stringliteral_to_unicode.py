import gc
import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
from numba.np import numpy_support
def test_stringliteral_to_unicode(self):

    @jit(types.void(types.unicode_type), nopython=True)
    def bar(string):
        pass

    @jit(types.void(), nopython=True)
    def foo2():
        bar('literal string')