import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import opcode
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase
def test_stack_effects(self):
    from _pydevd_frame_eval.vendored.bytecode.concrete import ConcreteInstr

    def check(instr):
        jump = instr.stack_effect(jump=True)
        no_jump = instr.stack_effect(jump=False)
        max_effect = instr.stack_effect(jump=None)
        self.assertEqual(instr.stack_effect(), max_effect)
        self.assertEqual(max_effect, max(jump, no_jump))
        if not instr.has_jump():
            self.assertEqual(jump, no_jump)
    for name, op in opcode.opmap.items():
        with self.subTest(name):
            if op < opcode.HAVE_ARGUMENT:
                check(ConcreteInstr(name))
            else:
                for arg in range(256):
                    check(ConcreteInstr(name, arg))
    for arg in (2 ** 31, 2 ** 32, 2 ** 63, 2 ** 64, -1):
        self.assertEqual(Instr('LOAD_CONST', arg).stack_effect(), 1)