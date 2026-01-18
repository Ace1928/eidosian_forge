import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def test_nop(self):
    code = Bytecode([Instr('LOAD_NAME', 'x'), Instr('NOP'), Instr('STORE_NAME', 'test')])
    self.check(code, Instr('LOAD_NAME', 'x'), Instr('STORE_NAME', 'test'))