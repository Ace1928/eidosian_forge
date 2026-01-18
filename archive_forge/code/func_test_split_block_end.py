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
def test_split_block_end(self):
    code = self.sample_code()
    label = code.split_block(code[0], 2)
    self.assertIs(label, code[1])
    self.assertBlocksEqual(code, [Instr('LOAD_CONST', 1, lineno=1), Instr('STORE_NAME', 'x', lineno=1)], [])
    self.check_getitem(code)
    label = code.split_block(code[0], 2)
    self.assertIs(label, code[1])
    self.assertBlocksEqual(code, [Instr('LOAD_CONST', 1, lineno=1), Instr('STORE_NAME', 'x', lineno=1)], [])