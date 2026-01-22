import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
class TestTryExceptCaught(TestCase):

    def test_catch_exception(self):

        @njit
        def udt(x):
            try:
                print('A')
                if x:
                    raise ZeroDivisionError('321')
                print('B')
            except Exception:
                print('C')
            print('D')
        with captured_stdout() as stdout:
            udt(True)
        self.assertEqual(stdout.getvalue().split(), ['A', 'C', 'D'])
        with captured_stdout() as stdout:
            udt(False)
        self.assertEqual(stdout.getvalue().split(), ['A', 'B', 'D'])

    def test_return_in_catch(self):

        @njit
        def udt(x):
            try:
                print('A')
                if x:
                    raise ZeroDivisionError
                print('B')
                r = 123
            except Exception:
                print('C')
                r = 321
                return r
            print('D')
            return r
        with captured_stdout() as stdout:
            res = udt(True)
        self.assertEqual(stdout.getvalue().split(), ['A', 'C'])
        self.assertEqual(res, 321)
        with captured_stdout() as stdout:
            res = udt(False)
        self.assertEqual(stdout.getvalue().split(), ['A', 'B', 'D'])
        self.assertEqual(res, 123)

    def test_save_caught(self):

        @njit
        def udt(x):
            try:
                if x:
                    raise ZeroDivisionError
                r = 123
            except Exception as e:
                r = 321
                return r
            return r
        with self.assertRaises(UnsupportedError) as raises:
            udt(True)
        self.assertIn('Exception object cannot be stored into variable (e)', str(raises.exception))

    def test_try_except_reraise(self):

        @njit
        def udt():
            try:
                raise ValueError('ERROR')
            except Exception:
                raise
        with self.assertRaises(UnsupportedError) as raises:
            udt()
        self.assertIn('The re-raising of an exception is not yet supported.', str(raises.exception))

    def test_try_except_reraise_chain(self):

        @njit
        def udt():
            try:
                raise ValueError('ERROR')
            except Exception:
                try:
                    raise
                except Exception:
                    raise
        with self.assertRaises(UnsupportedError) as raises:
            udt()
        self.assertIn('The re-raising of an exception is not yet supported.', str(raises.exception))

    def test_division_operator(self):

        @njit
        def udt(y):
            try:
                1 / y
            except Exception:
                return 57005
            else:
                return 1 / y
        self.assertEqual(udt(0), 57005)
        self.assertEqual(udt(2), 0.5)