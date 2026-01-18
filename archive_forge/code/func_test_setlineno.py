import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import textwrap
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, FreeVar, Bytecode, SetLineno, ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, get_code
def test_setlineno(self):
    code = Bytecode()
    code.first_lineno = 3
    code.extend([Instr('LOAD_CONST', 7), Instr('STORE_NAME', 'x'), SetLineno(4), Instr('LOAD_CONST', 8), Instr('STORE_NAME', 'y'), SetLineno(5), Instr('LOAD_CONST', 9), Instr('STORE_NAME', 'z')])
    concrete = code.to_concrete_bytecode()
    self.assertEqual(concrete.consts, [7, 8, 9])
    self.assertEqual(concrete.names, ['x', 'y', 'z'])
    self.assertListEqual(list(concrete), [ConcreteInstr('LOAD_CONST', 0, lineno=3), ConcreteInstr('STORE_NAME', 0, lineno=3), ConcreteInstr('LOAD_CONST', 1, lineno=4), ConcreteInstr('STORE_NAME', 1, lineno=4), ConcreteInstr('LOAD_CONST', 2, lineno=5), ConcreteInstr('STORE_NAME', 2, lineno=5)])