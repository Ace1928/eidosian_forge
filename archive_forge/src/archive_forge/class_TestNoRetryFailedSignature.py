import multiprocessing
import platform
import threading
import pickle
import weakref
from itertools import chain
from io import StringIO
import numpy as np
from numba import njit, jit, typeof, vectorize
from numba.core import types, errors
from numba import _dispatcher
from numba.tests.support import TestCase, captured_stdout
from numba.np.numpy_support import as_dtype
from numba.core.dispatcher import Dispatcher
from numba.extending import overload
from numba.tests.support import needs_lapack, SerialMixin
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
import unittest
class TestNoRetryFailedSignature(unittest.TestCase):
    """Test that failed-to-compile signatures are not recompiled.
    """

    def run_test(self, func):
        fcom = func._compiler
        self.assertEqual(len(fcom._failed_cache), 0)
        with self.assertRaises(errors.TypingError):
            func(1)
        self.assertEqual(len(fcom._failed_cache), 1)
        with self.assertRaises(errors.TypingError):
            func(1)
        self.assertEqual(len(fcom._failed_cache), 1)
        with self.assertRaises(errors.TypingError):
            func(1.0)
        self.assertEqual(len(fcom._failed_cache), 2)

    def test_direct_call(self):

        @jit(nopython=True)
        def foo(x):
            return x[0]
        self.run_test(foo)

    def test_nested_call(self):

        @jit(nopython=True)
        def bar(x):
            return x[0]

        @jit(nopython=True)
        def foobar(x):
            bar(x)

        @jit(nopython=True)
        def foo(x):
            return bar(x) + foobar(x)
        self.run_test(foo)

    @unittest.expectedFailure
    def test_error_count(self):

        def check(field, would_fail):
            k = 10
            counter = {'c': 0}

            def trigger(x):
                assert 0, 'unreachable'

            @overload(trigger)
            def ol_trigger(x):
                counter['c'] += 1
                if would_fail:
                    raise errors.TypingError('invoke_failed')
                return lambda x: x

            @jit(nopython=True)
            def ident(out, x):
                pass

            def chain_assign(fs, inner=ident):
                tab_head, tab_tail = (fs[-1], fs[:-1])

                @jit(nopython=True)
                def assign(out, x):
                    inner(out, x)
                    out[0] += tab_head(x)
                if tab_tail:
                    return chain_assign(tab_tail, assign)
                else:
                    return assign
            chain = chain_assign((trigger,) * k)
            out = np.ones(2)
            if would_fail:
                with self.assertRaises(errors.TypingError) as raises:
                    chain(out, 1)
                self.assertIn('invoke_failed', str(raises.exception))
            else:
                chain(out, 1)
            return counter['c']
        ct_ok = check('a', False)
        ct_bad = check('c', True)
        self.assertEqual(ct_ok, 1)
        self.assertEqual(ct_bad, 1)