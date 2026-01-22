import sys
import warnings
import numpy as np
from numba import njit, literally
from numba import int32, int64, float32, float64
from numba import typeof
from numba.typed import Dict, dictobject, List
from numba.typed.typedobjectutils import _sentry_safe_cast
from numba.core.errors import TypingError
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin, unittest,
from numba.experimental import jitclass
from numba.extending import overload
class TestDictIterableCtorNoJit(TestCase, DictIterableCtor):

    def setUp(self):
        self.jit_enabled = False

    def test_exception_nargs(self):
        msg = 'Dict expect at most 1 argument, got 2'
        with self.assertRaisesRegex(TypingError, msg):
            Dict(1, 2)

    def test_exception_mapping_ctor(self):
        msg = '.*dict\\(mapping\\) is not supported.*'
        with self.assertRaisesRegex(TypingError, msg):
            Dict({1: 2})

    def test_exception_non_iterable_arg(self):
        msg = '.*object is not iterable.*'
        with self.assertRaisesRegex(TypingError, msg):
            Dict(3)

    def test_exception_setitem(self):
        msg = '.*dictionary update sequence element #1 has length 3.*'
        with self.assertRaisesRegex(ValueError, msg):
            Dict(((1, 'a'), (2, 'b', 3)))