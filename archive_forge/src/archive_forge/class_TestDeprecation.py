import warnings
import unittest
from contextlib import contextmanager
from numba import jit, vectorize, guvectorize
from numba.core.errors import (NumbaDeprecationWarning,
from numba.tests.support import TestCase, needs_setuptools
class TestDeprecation(TestCase):

    def check_warning(self, warnings, expected_str, category, check_rtd=True):
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].category, category)
        self.assertIn(expected_str, str(warnings[0].message))
        if check_rtd:
            self.assertIn('https://numba.readthedocs.io', str(warnings[0].message))

    @TestCase.run_test_in_subprocess
    def test_explicit_false_nopython_kwarg(self):
        with _catch_numba_deprecation_warnings() as w:

            @jit(nopython=False)
            def foo():
                pass
            foo()
            msg = "The keyword argument 'nopython=False' was supplied"
            self.check_warning(w, msg, NumbaDeprecationWarning, check_rtd=False)

    @TestCase.run_test_in_subprocess
    def test_vectorize_missing_nopython_kwarg_not_reported(self):
        with _catch_numba_deprecation_warnings() as w:

            @vectorize('float64(float64)')
            def foo(a):
                return a + 1
        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_vectorize_nopython_false_is_reported(self):
        with _catch_numba_deprecation_warnings() as w:

            @vectorize('float64(float64)', nopython=False)
            def foo(a):
                return a + 1
        msg = "The keyword argument 'nopython=False' was supplied"
        self.check_warning(w, msg, NumbaDeprecationWarning, check_rtd=False)

    @TestCase.run_test_in_subprocess
    def test_vectorize_objmode_direct_compilation_no_warnings(self):
        with _catch_numba_deprecation_warnings() as w:

            @vectorize('float64(float64)', forceobj=True)
            def foo(a):
                object()
                return a + 1
        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_vectorize_objmode_compilation_nopython_false_warns(self):
        with _catch_numba_deprecation_warnings() as w:

            @vectorize('float64(float64)', forceobj=True, nopython=False)
            def foo(a):
                object()
                return a + 1
        msg = "The keyword argument 'nopython=False' was supplied"
        self.check_warning(w, msg, NumbaDeprecationWarning, check_rtd=False)

    @TestCase.run_test_in_subprocess
    def test_vectorize_parallel_true_no_warnings(self):
        with _catch_numba_deprecation_warnings() as w:

            @vectorize('float64(float64)', target='parallel')
            def foo(x):
                return x + 1
        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_vectorize_parallel_true_nopython_true_no_warnings(self):
        with _catch_numba_deprecation_warnings() as w:

            @vectorize('float64(float64)', target='parallel', nopython=True)
            def foo(x):
                return x + 1
        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_vectorize_parallel_true_nopython_false_warns(self):
        with _catch_numba_deprecation_warnings() as w:

            @vectorize('float64(float64)', target='parallel', nopython=False)
            def foo(x):
                return x + 1
        msg = "The keyword argument 'nopython=False' was supplied"
        self.check_warning(w, msg, NumbaDeprecationWarning, check_rtd=False)

    @TestCase.run_test_in_subprocess
    def test_vectorize_calling_jit_with_nopython_false_warns_from_jit(self):
        with _catch_numba_deprecation_warnings() as w:

            @vectorize('float64(float64)', forceobj=True)
            def foo(x):
                return bar(x + 1)

            def bar(*args):
                pass
        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_guvectorize_implicit_nopython_no_warnings(self):
        with _catch_numba_deprecation_warnings() as w:

            @guvectorize('void(float64[::1], float64[::1])', '(n)->(n)')
            def bar(a, b):
                a += 1
        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_guvectorize_forceobj_no_warnings(self):
        with _catch_numba_deprecation_warnings() as w:

            @guvectorize('void(float64[::1], float64[::1])', '(n)->(n)', forceobj=True)
            def bar(a, b):
                object()
                a += 1
        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_guvectorize_parallel_implicit_nopython_no_warnings(self):
        with _catch_numba_deprecation_warnings() as w:

            @guvectorize('void(float64[::1], float64[::1])', '(n)->(n)', target='parallel')
            def bar(a, b):
                a += 1
        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_guvectorize_parallel_forceobj_no_warnings(self):
        with _catch_numba_deprecation_warnings() as w:

            @guvectorize('void(float64[::1], float64[::1])', '(n)->(n)', target='parallel', forceobj=True)
            def bar(a, b):
                object()
                a += 1
        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_reflection_of_mutable_container(self):

        def foo_list(a):
            return a.append(1)

        def foo_set(a):
            return a.add(1)
        for f in [foo_list, foo_set]:
            container = f.__name__.strip('foo_')
            inp = eval(container)([10])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('ignore', category=NumbaWarning)
                warnings.simplefilter('always', category=NumbaPendingDeprecationWarning)
                jit(nopython=True)(f)(inp)
                self.assertEqual(len(w), 1)
                self.assertEqual(w[0].category, NumbaPendingDeprecationWarning)
                warn_msg = str(w[0].message)
                msg = 'Encountered the use of a type that is scheduled for deprecation'
                self.assertIn(msg, warn_msg)
                msg = "'reflected %s' found for argument" % container
                self.assertIn(msg, warn_msg)
                self.assertIn('https://numba.readthedocs.io', warn_msg)

    @needs_setuptools
    @TestCase.run_test_in_subprocess
    def test_pycc_module(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', category=NumbaPendingDeprecationWarning)
            import numba.pycc
            expected_str = "The 'pycc' module is pending deprecation."
            self.check_warning(w, expected_str, NumbaPendingDeprecationWarning)

    @needs_setuptools
    @TestCase.run_test_in_subprocess
    def test_pycc_CC(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', category=NumbaPendingDeprecationWarning)
            from numba.pycc import CC
            expected_str = "The 'pycc' module is pending deprecation."
            self.check_warning(w, expected_str, NumbaPendingDeprecationWarning)