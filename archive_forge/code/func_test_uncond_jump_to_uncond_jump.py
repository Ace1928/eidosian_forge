import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def test_uncond_jump_to_uncond_jump(self):
    label = Label()
    label2 = Label()
    label3 = Label()
    label4 = Label()
    code = Bytecode([Instr('LOAD_NAME', 'test'), Instr('POP_JUMP_IF_TRUE', label), Instr('JUMP_FORWARD', label2), label, Instr('LOAD_CONST', 1), Instr('STORE_NAME', 'x'), Instr('LOAD_NAME', 'test'), Instr('POP_JUMP_IF_TRUE', label3), label2, Instr('JUMP_FORWARD', label4), label3, Instr('LOAD_CONST', 1), Instr('STORE_NAME', 'x'), label4, Instr('LOAD_CONST', None), Instr('RETURN_VALUE')])
    label = Label()
    label3 = Label()
    label4 = Label()
    self.check(code, Instr('LOAD_NAME', 'test'), Instr('POP_JUMP_IF_TRUE', label), Instr('JUMP_ABSOLUTE', label4), label, Instr('LOAD_CONST', 1), Instr('STORE_NAME', 'x'), Instr('LOAD_NAME', 'test'), Instr('POP_JUMP_IF_TRUE', label3), Instr('JUMP_FORWARD', label4), label3, Instr('LOAD_CONST', 1), Instr('STORE_NAME', 'x'), label4, Instr('LOAD_CONST', None), Instr('RETURN_VALUE'))