import itertools
import functools
import sys
import operator
from collections import namedtuple
import numpy as np
import unittest
import warnings
from numba import jit, typeof, njit, typed
from numba.core import errors, types, config
from numba.tests.support import (TestCase, tag, ignore_internal_warnings,
from numba.core.extending import overload_method, box
def test_numba_types(self):

    def gen_w_arg(clazz_type):

        def impl():
            return isinstance(1, clazz_type)
        return impl
    clazz_types = (types.Integer, types.Float, types.Array)
    msg = 'Numba type classes.*are not supported'
    for ct in clazz_types:
        fn = njit(gen_w_arg(ct))
        with self.assertRaises(errors.TypingError) as raises:
            fn()
        self.assertRegex(str(raises.exception), msg)