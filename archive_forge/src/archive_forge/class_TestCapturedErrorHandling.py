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
class TestCapturedErrorHandling(SerialMixin, TestCase):
    """Checks that the way errors are captured changes depending on the env
    var "NUMBA_CAPTURED_ERRORS".
    """

    def test_error_in_overload(self):

        def bar(x):
            pass

        @overload(bar)
        def ol_bar(x):
            x.some_invalid_attr

            def impl(x):
                pass
            return impl
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', errors.NumbaPendingDeprecationWarning)
            for style, err_class in (('new_style', AttributeError), ('old_style', errors.TypingError)):
                with override_config('CAPTURED_ERRORS', style):
                    with self.assertRaises(err_class) as raises:

                        @njit('void(int64)')
                        def foo(x):
                            bar(x)
                    expected = "object has no attribute 'some_invalid_attr'"
                    self.assertIn(expected, str(raises.exception))

    def _run_in_separate_process(self, runcode, env):
        code = f'if 1:\n            {runcode}\n\n            '
        proc_env = os.environ.copy()
        proc_env.update(env)
        popen = subprocess.Popen([sys.executable, '-Wall', '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=proc_env)
        out, err = popen.communicate()
        if popen.returncode != 0:
            raise AssertionError('process failed with code %s: stderr follows\n%s\n' % (popen.returncode, err.decode()))
        return (out, err)

    def test_old_style_deprecation_on_import(self):
        from numba.core.config import _old_style_deprecation_msg
        code = '\n        import numba\n        '
        env = {'NUMBA_CAPTURED_ERRORS': 'old_style'}
        _out, err = self._run_in_separate_process(code, env)
        self.assertIn(_old_style_deprecation_msg, err.decode())
        env = {'NUMBA_CAPTURED_ERRORS': ''}
        _out, err = self._run_in_separate_process(code, env)
        self.assertNotIn('NumbaPendingDeprecationWarning', err.decode())
        env = {'NUMBA_CAPTURED_ERRORS': 'new_style'}
        _out, err = self._run_in_separate_process(code, env)
        self.assertNotIn('NumbaPendingDeprecationWarning', err.decode())

    def _test_old_style_deprecation(self):
        warnings.simplefilter('always', errors.NumbaPendingDeprecationWarning)

        def bar(x):
            pass

        @overload(bar)
        def ol_bar(x):
            raise AttributeError('Invalid attribute')
        with self.assertWarns(errors.NumbaPendingDeprecationWarning) as warns:
            with self.assertRaises(errors.TypingError):

                @njit('void(int64)')
                def foo(x):
                    bar(x)
            self.assertIn("Code using Numba extension API maybe depending on 'old_style' error-capturing", str(warns.warnings[0].message))
    test_old_style_deprecation = TestCase.run_test_in_subprocess(envvars={'NUMBA_CAPTURED_ERRORS': 'old_style'})(_test_old_style_deprecation)
    test_default_old_style_deprecation = TestCase.run_test_in_subprocess(envvars={'NUMBA_CAPTURED_ERRORS': 'default'})(_test_old_style_deprecation)

    @TestCase.run_test_in_subprocess(envvars={'NUMBA_CAPTURED_ERRORS': 'old_style'})
    def test_old_style_no_deprecation(self):
        warnings.simplefilter('always', errors.NumbaPendingDeprecationWarning)

        def bar(x):
            pass

        @overload(bar)
        def ol_bar(x):
            raise errors.TypingError('Invalid attribute')
        with warnings.catch_warnings(record=True) as warns:
            with self.assertRaises(errors.TypingError):

                @njit('void(int64)')
                def foo(x):
                    bar(x)
            self.assertEqual(len(warns), 0, msg='There should not be any warnings')

    @TestCase.run_test_in_subprocess(envvars={'NUMBA_CAPTURED_ERRORS': 'new_style'})
    def test_new_style_no_warnings(self):
        warnings.simplefilter('always', errors.NumbaPendingDeprecationWarning)

        def bar(x):
            pass

        @overload(bar)
        def ol_bar(x):
            raise AttributeError('Invalid attribute')
        with warnings.catch_warnings(record=True) as warns:
            with self.assertRaises(AttributeError):

                @njit('void(int64)')
                def foo(x):
                    bar(x)
            self.assertEqual(len(warns), 0, msg='There should not be any warnings')