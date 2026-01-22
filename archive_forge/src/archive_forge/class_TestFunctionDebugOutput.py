import os
import platform
import re
import textwrap
import warnings
import numpy as np
from numba.tests.support import (TestCase, override_config, override_env_config,
from numba import jit, njit
from numba.core import types, compiler, utils
from numba.core.errors import NumbaPerformanceWarning
from numba import prange
from numba.experimental import jitclass
import unittest
class TestFunctionDebugOutput(FunctionDebugTestBase):

    def test_dump_bytecode(self):
        with override_config('DUMP_BYTECODE', True):
            out = self.compile_simple_nopython()
        self.check_debug_output(out, ['bytecode'])

    def test_dump_ir(self):
        with override_config('DUMP_IR', True):
            out = self.compile_simple_nopython()
        self.check_debug_output(out, ['ir'])

    def test_dump_cfg(self):
        with override_config('DUMP_CFG', True):
            out = self.compile_simple_nopython()
        self.check_debug_output(out, ['cfg'])

    def test_dump_llvm(self):
        with override_config('DUMP_LLVM', True):
            out = self.compile_simple_nopython()
        self.check_debug_output(out, ['llvm'])

    def test_dump_func_opt_llvm(self):
        with override_config('DUMP_FUNC_OPT', True):
            out = self.compile_simple_nopython()
        self.check_debug_output(out, ['func_opt_llvm'])

    def test_dump_optimized_llvm(self):
        with override_config('DUMP_OPTIMIZED', True):
            out = self.compile_simple_nopython()
        self.check_debug_output(out, ['optimized_llvm'])

    def test_dump_assembly(self):
        with override_config('DUMP_ASSEMBLY', True):
            out = self.compile_simple_nopython()
        self.check_debug_output(out, ['assembly'])