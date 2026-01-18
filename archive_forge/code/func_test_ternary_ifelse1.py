import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_ternary_ifelse1(self, flags=enable_pyobj_flags):
    self.run_test(ternary_ifelse_usecase1, [-1, 0, 1], [-1, 0, 1], flags=flags)