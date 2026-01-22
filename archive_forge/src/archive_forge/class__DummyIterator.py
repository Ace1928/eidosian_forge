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
class _DummyIterator(_Dummy):
    count = 0

    def __next__(self):
        if self.count >= 3:
            raise StopIteration
        self.count += 1
        return _Dummy(self.recorder, '%s#%s' % (self.name, self.count))
    next = __next__