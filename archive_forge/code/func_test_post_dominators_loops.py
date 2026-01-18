import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_post_dominators_loops(self):
    g = self.multiple_exits()
    doms = g.post_dominators()
    self.check_dominators(doms, {0: [0, 7], 7: [7], 10: [10], 19: [19], 23: [23], 29: [29, 37], 36: [36, 37], 37: [37]})
    g = self.multiple_loops()
    doms = g.post_dominators()
    self.check_dominators(doms, {0: [0, 60, 68, 61, 7], 7: [60, 68, 61, 7], 10: [68, 7, 10, 13, 20, 56, 57, 60, 61], 13: [68, 7, 13, 20, 56, 57, 60, 61], 20: [20, 68, 7, 56, 57, 60, 61], 23: [68, 7, 20, 23, 56, 57, 60, 61], 32: [32, 68, 7, 20, 56, 57, 60, 61], 44: [68, 7, 44, 20, 56, 57, 60, 61], 56: [68, 7, 56, 57, 60, 61], 57: [57, 60, 68, 61, 7], 60: [60, 68, 61], 61: [68, 61], 68: [68], 71: [71], 80: [80], 87: [88, 87], 88: [88]})