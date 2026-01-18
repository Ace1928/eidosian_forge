import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import io
import sys
import unittest
import contextlib
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.concrete import OFFSET_AS_INSTRUCTION
from _pydevd_frame_eval.vendored.bytecode.tests import disassemble as _disassemble, TestCase
def test_to_bytecode(self):
    blocks = ControlFlowGraph()
    blocks.add_block()
    blocks.add_block()
    blocks[0].extend([Instr('LOAD_NAME', 'test', lineno=1), Instr('POP_JUMP_IF_FALSE', blocks[2], lineno=1)])
    blocks[1].extend([Instr('LOAD_CONST', 5, lineno=2), Instr('STORE_NAME', 'x', lineno=2), Instr('JUMP_FORWARD', blocks[2], lineno=2)])
    blocks[2].extend([Instr('LOAD_CONST', 7, lineno=3), Instr('STORE_NAME', 'x', lineno=3), Instr('LOAD_CONST', None, lineno=3), Instr('RETURN_VALUE', lineno=3)])
    bytecode = blocks.to_bytecode()
    label = Label()
    self.assertEqual(bytecode, [Instr('LOAD_NAME', 'test', lineno=1), Instr('POP_JUMP_IF_FALSE', label, lineno=1), Instr('LOAD_CONST', 5, lineno=2), Instr('STORE_NAME', 'x', lineno=2), Instr('JUMP_FORWARD', label, lineno=2), label, Instr('LOAD_CONST', 7, lineno=3), Instr('STORE_NAME', 'x', lineno=3), Instr('LOAD_CONST', None, lineno=3), Instr('RETURN_VALUE', lineno=3)])