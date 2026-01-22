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
class TestUfuncOnContext(TestCase):

    def test_cpu_get_ufunc_info(self):
        targetctx = cpu_target.target_context
        add_info = targetctx.get_ufunc_info(np.add)
        self.assertIsInstance(add_info, dict)
        expected = ufunc_db.get_ufunc_info(np.add)
        self.assertEqual(add_info, expected)
        badkey = object()
        with self.assertRaises(KeyError) as raises:
            ufunc_db.get_ufunc_info(badkey)
        self.assertEqual(raises.exception.args, (badkey,))

    def test_base_get_ufunc_info(self):
        targetctx = BaseContext(cpu_target.typing_context, 'cpu')
        with self.assertRaises(NotImplementedError) as raises:
            targetctx.get_ufunc_info(np.add)
        self.assertRegex(str(raises.exception), '<numba\\..*\\.BaseContext object at .*> does not support ufunc')