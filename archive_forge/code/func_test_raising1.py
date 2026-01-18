import collections
import weakref
import gc
import operator
from itertools import takewhile
import unittest
from numba import njit, jit
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.untyped_passes import PreserveIR
from numba.core.typed_passes import IRLegalization
from numba.core import types, ir
from numba.tests.support import TestCase, override_config, SerialMixin
def test_raising1(self):
    with self.assertRefCount(do_raise):
        rec = self.compile_and_record(raising_usecase1, raises=MyError)
        self.assertFalse(rec.alive)