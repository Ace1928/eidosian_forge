import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import textwrap
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, FreeVar, Bytecode, SetLineno, ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, get_code
def test_not_enough_dup(self):
    code = Bytecode()
    code.first_lineno = 1
    code.extend([Instr('LOAD_CONST', 1), Instr('DUP_TOP_TWO')])
    with self.assertRaises(RuntimeError):
        code.compute_stacksize()