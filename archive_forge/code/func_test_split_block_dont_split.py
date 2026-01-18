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
def test_split_block_dont_split(self):
    code = self.sample_code()
    block = code.split_block(code[0], 0)
    self.assertIs(block, code[0])
    self.assertBlocksEqual(code, [Instr('LOAD_CONST', 1, lineno=1), Instr('STORE_NAME', 'x', lineno=1)])