import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import unittest
from _pydevd_frame_eval.vendored.bytecode import ConcreteBytecode, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode.tests import get_code
Check that bytecode.from_code(code).to_code() returns code.