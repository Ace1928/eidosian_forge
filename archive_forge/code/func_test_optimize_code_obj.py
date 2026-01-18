import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def test_optimize_code_obj(self):
    noopt = Bytecode([Instr('LOAD_CONST', 3), Instr('LOAD_CONST', 5), Instr('BINARY_ADD'), Instr('STORE_NAME', 'x'), Instr('LOAD_CONST', None), Instr('RETURN_VALUE')])
    noopt = noopt.to_code()
    optimizer = peephole_opt.PeepholeOptimizer()
    optim = optimizer.optimize(noopt)
    code = Bytecode.from_code(optim)
    self.assertEqual(code, [Instr('LOAD_CONST', 8, lineno=1), Instr('STORE_NAME', 'x', lineno=1), Instr('LOAD_CONST', None, lineno=1), Instr('RETURN_VALUE', lineno=1)])