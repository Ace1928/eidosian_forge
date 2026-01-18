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
def test_stack_size_computation_for_loop_continue(self):

    def test(arg1, *args, **kwargs):
        for k in kwargs:
            if k in args:
                continue
        else:
            return 1
    self.check_stack_size(test)