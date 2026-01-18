import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def test_unconditional_jump_to_return(self):
    label_instr11 = Label()
    label_instr14 = Label()
    label_instr7 = Label()
    code = Bytecode([Instr('LOAD_GLOBAL', 'test', lineno=2), Instr('POP_JUMP_IF_FALSE', label_instr11, lineno=2), Instr('LOAD_GLOBAL', 'test2', lineno=3), Instr('POP_JUMP_IF_FALSE', label_instr7, lineno=3), Instr('LOAD_CONST', 10, lineno=4), Instr('STORE_FAST', 'x', lineno=4), Instr('JUMP_ABSOLUTE', label_instr14, lineno=4), label_instr7, Instr('LOAD_CONST', 20, lineno=6), Instr('STORE_FAST', 'x', lineno=6), Instr('JUMP_FORWARD', label_instr14, lineno=6), label_instr11, Instr('LOAD_CONST', 30, lineno=8), Instr('STORE_FAST', 'x', lineno=8), label_instr14, Instr('LOAD_CONST', None, lineno=8), Instr('RETURN_VALUE', lineno=8)])
    label1 = Label()
    label3 = Label()
    label4 = Label()
    self.check(code, Instr('LOAD_GLOBAL', 'test', lineno=2), Instr('POP_JUMP_IF_FALSE', label3, lineno=2), Instr('LOAD_GLOBAL', 'test2', lineno=3), Instr('POP_JUMP_IF_FALSE', label1, lineno=3), Instr('LOAD_CONST', 10, lineno=4), Instr('STORE_FAST', 'x', lineno=4), Instr('JUMP_ABSOLUTE', label4, lineno=4), label1, Instr('LOAD_CONST', 20, lineno=6), Instr('STORE_FAST', 'x', lineno=6), Instr('JUMP_FORWARD', label4, lineno=6), label3, Instr('LOAD_CONST', 30, lineno=8), Instr('STORE_FAST', 'x', lineno=8), label4, Instr('LOAD_CONST', None, lineno=8), Instr('RETURN_VALUE', lineno=8))