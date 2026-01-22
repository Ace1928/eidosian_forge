import sys
import subprocess
import numpy as np
import os
import warnings
from numba import jit, njit, types
from numba.core import errors
from numba.experimental import structref
from numba.extending import (overload, intrinsic, overload_method,
from numba.core.compiler import CompilerBase
from numba.core.untyped_passes import (TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, DeadCodeElimination,
from numba.core.compiler_machinery import PassManager
from numba.core.types.functions import _err_reasons as error_reasons
from numba.tests.support import (skip_parfors_unsupported, override_config,
import unittest
class TestMiscErrorHandling(unittest.TestCase):

    def test_use_of_exception_for_flow_control(self):

        @njit
        def fn(x):
            return 10 ** x
        a = np.array([1.0], dtype=np.float64)
        fn(a)

    def test_commented_func_definition_is_not_a_definition(self):

        def foo_commented():
            raise Exception('test_string')

        def foo_docstring():
            """ def docstring containing def might match function definition!"""
            raise Exception('test_string')
        for func in (foo_commented, foo_docstring):
            with self.assertRaises(Exception) as raises:
                func()
            self.assertIn('test_string', str(raises.exception))

    def test_use_of_ir_unknown_loc(self):

        class TestPipeline(CompilerBase):

            def define_pipelines(self):
                name = 'bad_DCE_pipeline'
                pm = PassManager(name)
                pm.add_pass(TranslateByteCode, 'analyzing bytecode')
                pm.add_pass(FixupArgs, 'fix up args')
                pm.add_pass(IRProcessing, 'processing IR')
                pm.add_pass(DeadCodeElimination, 'DCE')
                pm.add_pass(NopythonTypeInference, 'nopython frontend')
                pm.add_pass(NativeLowering, 'native lowering')
                pm.add_pass(NoPythonBackend, 'nopython mode backend')
                pm.finalize()
                return [pm]

        @njit(pipeline_class=TestPipeline)
        def f(a):
            return 0
        with self.assertRaises(errors.TypingError) as raises:
            f(iter([1, 2]))
        expected = 'File "unknown location", line 0:'
        self.assertIn(expected, str(raises.exception))

    def check_write_to_globals(self, func):
        with self.assertRaises(errors.TypingError) as raises:
            func()
        expected = ['The use of a', 'in globals, is not supported as globals']
        for ex in expected:
            self.assertIn(ex, str(raises.exception))

    def test_handling_of_write_to_reflected_global(self):
        from numba.tests.errorhandling_usecases import global_reflected_write
        self.check_write_to_globals(njit(global_reflected_write))

    def test_handling_of_write_to_typed_dict_global(self):
        from numba.tests.errorhandling_usecases import global_dict_write
        self.check_write_to_globals(njit(global_dict_write))

    @skip_parfors_unsupported
    def test_handling_forgotten_numba_internal_import(self):

        @njit(parallel=True)
        def foo():
            for i in prange(10):
                pass
        with self.assertRaises(errors.TypingError) as raises:
            foo()
        expected = "'prange' looks like a Numba internal function, has it been imported"
        self.assertIn(expected, str(raises.exception))

    def test_handling_unsupported_generator_expression(self):

        def foo():
            (x for x in range(10))
        expected = 'The use of yield in a closure is unsupported.'
        for dec in (jit(forceobj=True), njit):
            with self.assertRaises(errors.UnsupportedError) as raises:
                dec(foo)()
            self.assertIn(expected, str(raises.exception))

    def test_handling_undefined_variable(self):

        @njit
        def foo():
            return a
        expected = "NameError: name 'a' is not defined"
        with self.assertRaises(errors.TypingError) as raises:
            foo()
        self.assertIn(expected, str(raises.exception))