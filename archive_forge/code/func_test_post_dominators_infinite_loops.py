import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_post_dominators_infinite_loops(self):
    g = self.infinite_loop1()
    doms = g.post_dominators()
    self.check_dominators(doms, {0: [0], 6: [6], 10: [10, 13], 13: [13], 19: [19], 26: [26]})
    g = self.infinite_loop2()
    doms = g.post_dominators()
    self.check_dominators(doms, {0: [0, 3], 3: [3], 9: [9], 16: [16]})