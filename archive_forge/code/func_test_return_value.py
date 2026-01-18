import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def test_return_value(self):
    code = Bytecode([Instr('LOAD_CONST', 4, lineno=2), Instr('RETURN_VALUE', lineno=2), Instr('LOAD_CONST', 5, lineno=3), Instr('RETURN_VALUE', lineno=3)])
    code = ControlFlowGraph.from_bytecode(code)
    self.check(code, Instr('LOAD_CONST', 4, lineno=2), Instr('RETURN_VALUE', lineno=2))
    code = Bytecode([Instr('LOAD_CONST', 4, lineno=2), Instr('RETURN_VALUE', lineno=2), Instr('LOAD_CONST', 5, lineno=3), Instr('RETURN_VALUE', lineno=3), Instr('LOAD_CONST', 6, lineno=4), Instr('RETURN_VALUE', lineno=4), Instr('LOAD_CONST', 7, lineno=5), Instr('RETURN_VALUE', lineno=5)])
    code = ControlFlowGraph.from_bytecode(code)
    self.check(code, Instr('LOAD_CONST', 4, lineno=2), Instr('RETURN_VALUE', lineno=2))
    if sys.version_info < (3, 8):
        setup_loop = Label()
        return_label = Label()
        code = Bytecode([setup_loop, Instr('SETUP_LOOP', return_label, lineno=2), Instr('LOAD_CONST', 7, lineno=3), Instr('RETURN_VALUE', lineno=3), Instr('JUMP_ABSOLUTE', setup_loop, lineno=3), Instr('POP_BLOCK', lineno=3), return_label, Instr('LOAD_CONST', None, lineno=3), Instr('RETURN_VALUE', lineno=3)])
        code = ControlFlowGraph.from_bytecode(code)
        end_loop = Label()
        self.check(code, Instr('SETUP_LOOP', end_loop, lineno=2), Instr('LOAD_CONST', 7, lineno=3), Instr('RETURN_VALUE', lineno=3), end_loop, Instr('LOAD_CONST', None, lineno=3), Instr('RETURN_VALUE', lineno=3))
    else:
        setup_loop = Label()
        return_label = Label()
        code = Bytecode([setup_loop, Instr('LOAD_CONST', 7, lineno=3), Instr('RETURN_VALUE', lineno=3), Instr('JUMP_ABSOLUTE', setup_loop, lineno=3), Instr('LOAD_CONST', None, lineno=3), Instr('RETURN_VALUE', lineno=3)])
        code = ControlFlowGraph.from_bytecode(code)
        self.check(code, Instr('LOAD_CONST', 7, lineno=3), Instr('RETURN_VALUE', lineno=3))