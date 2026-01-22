import functools
import itertools
import sys
import warnings
import threading
import operator
import numpy as np
import unittest
from numba import guvectorize, njit, typeof, vectorize
from numba.core import types
from numba.np.numpy_support import from_dtype
from numba.core.errors import LoweringError, TypingError
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.typing.npydecl import supported_ufuncs
from numba.np import numpy_support
from numba.core.registry import cpu_target
from numba.core.base import BaseContext
from numba.np import ufunc_db
@classmethod
def _inject_test(cls, ufunc, loop):

    def test_template(self):
        fn = _make_ufunc_usecase(ufunc)
        self._check_loop(fn, ufunc, loop)
    setattr(cls, 'test_{0}_{1}'.format(ufunc.__name__, loop.replace('->', '_')), test_template)