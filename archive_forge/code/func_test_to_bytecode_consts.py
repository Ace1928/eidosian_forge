import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import opcode
import sys
import textwrap
import types
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.concrete import OFFSET_AS_INSTRUCTION
from _pydevd_frame_eval.vendored.bytecode.tests import get_code, TestCase
def test_to_bytecode_consts(self):
    code = ConcreteBytecode()
    code.consts = [0.0, None, -0.0, 0.0]
    code.names = ['x', 'y']
    code.extend([ConcreteInstr('LOAD_CONST', 2, lineno=1), ConcreteInstr('STORE_NAME', 0, lineno=1), ConcreteInstr('LOAD_CONST', 3, lineno=2), ConcreteInstr('STORE_NAME', 1, lineno=2), ConcreteInstr('LOAD_CONST', 1, lineno=2), ConcreteInstr('RETURN_VALUE', lineno=2)])
    code = code.to_bytecode().to_concrete_bytecode()
    self.assertEqual(code.consts, [-0.0, 0.0, None])
    code.names = ['x', 'y']
    self.assertListEqual(list(code), [ConcreteInstr('LOAD_CONST', 0, lineno=1), ConcreteInstr('STORE_NAME', 0, lineno=1), ConcreteInstr('LOAD_CONST', 1, lineno=2), ConcreteInstr('STORE_NAME', 1, lineno=2), ConcreteInstr('LOAD_CONST', 2, lineno=2), ConcreteInstr('RETURN_VALUE', lineno=2)])