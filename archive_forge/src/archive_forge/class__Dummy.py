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
class _Dummy(object):

    def __init__(self, recorder, name):
        self.recorder = recorder
        self.name = name
        recorder._add_dummy(self)

    def __add__(self, other):
        assert isinstance(other, _Dummy)
        return _Dummy(self.recorder, '%s + %s' % (self.name, other.name))

    def __iter__(self):
        return _DummyIterator(self.recorder, 'iter(%s)' % self.name)