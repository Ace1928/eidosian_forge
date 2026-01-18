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
def test_simple1(self):
    rec = self.compile_and_record(simple_usecase1)
    self.assertFalse(rec.alive)
    self.assertRecordOrder(rec, ['a', 'b', '--1--'])
    self.assertRecordOrder(rec, ['a', 'c', '--1--'])
    self.assertRecordOrder(rec, ['--1--', 'b + c', '--2--'])