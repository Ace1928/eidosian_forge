import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_topo_sort(self):

    def check_topo_sort(nodes, expected):
        self.assertIn(list(g.topo_sort(nodes)), expected)
        self.assertIn(list(g.topo_sort(nodes[::-1])), expected)
        self.assertIn(list(g.topo_sort(nodes, reverse=True))[::-1], expected)
        self.assertIn(list(g.topo_sort(nodes[::-1], reverse=True))[::-1], expected)
        self.random.shuffle(nodes)
        self.assertIn(list(g.topo_sort(nodes)), expected)
        self.assertIn(list(g.topo_sort(nodes, reverse=True))[::-1], expected)
    g = self.loopless2()
    check_topo_sort([21, 99, 12, 34], ([99, 12, 21, 34],))
    check_topo_sort([18, 12, 42, 99], ([99, 12, 18, 42], [99, 18, 12, 42]))
    g = self.multiple_exits()
    check_topo_sort([19, 10, 7, 36], ([7, 10, 19, 36], [7, 10, 36, 19], [7, 36, 10, 19]))