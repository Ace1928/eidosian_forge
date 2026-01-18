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
def test_stack_size_computation_nested_try_except_finally(self):

    def test(arg1, *args, **kwargs):
        k = 1
        try:
            getattr(arg1, k)
        except AttributeError:
            pass
        except Exception:
            try:
                assert False
            except Exception:
                return 2
            finally:
                print('unexpected')
        finally:
            print('attempted to get {}'.format(k))
    self.check_stack_size(test)