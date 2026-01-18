import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_post_dominators_loopless(self):

    def eq_(d, l):
        self.assertEqual(sorted(doms[d]), l)
    for g in [self.loopless1(), self.loopless1_dead_nodes()]:
        doms = g.post_dominators()
        eq_(0, [0, 21])
        eq_(12, [12, 21])
        eq_(18, [18, 21])
        eq_(21, [21])
    g = self.loopless2()
    doms = g.post_dominators()
    eq_(34, [34])
    eq_(42, [42])
    eq_(21, [21])
    eq_(18, [18, 21])
    eq_(12, [12, 21])
    eq_(99, [21, 99])