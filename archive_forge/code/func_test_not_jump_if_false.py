import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def test_not_jump_if_false(self):
    label = Label()
    code = Bytecode([Instr('LOAD_NAME', 'x'), Instr('UNARY_NOT'), Instr('POP_JUMP_IF_FALSE', label), Instr('LOAD_CONST', 9), Instr('STORE_NAME', 'y'), label])
    code = self.optimize_blocks(code)
    label = Label()
    self.check(code, Instr('LOAD_NAME', 'x'), Instr('POP_JUMP_IF_TRUE', label), Instr('LOAD_CONST', 9), Instr('STORE_NAME', 'y'), label)