import sys
import unittest
from numba.tests.support import captured_stdout
from numba.core.config import IS_WIN32
def test_vectorize_dynamic(self):
    with captured_stdout():
        from numba import vectorize

        @vectorize
        def f(x, y):
            return x * y
        result = f(3, 4)
        print(f.types)
        self.assertEqual(result, 12)
        if IS_WIN32:
            correct = ['ll->q']
        else:
            correct = ['ll->l']
        self.assertEqual(f.types, correct)
        result = f(1.0, 2.0)
        print(f.types)
        self.assertEqual(result, 2.0)
        if IS_WIN32:
            correct = ['ll->q', 'dd->d']
        else:
            correct = ['ll->l', 'dd->d']
        self.assertEqual(f.types, correct)
        result = f(1, 2.0)
        print(f.types)
        self.assertEqual(result, 2.0)
        if IS_WIN32:
            correct = ['ll->q', 'dd->d']
        else:
            correct = ['ll->l', 'dd->d']
        self.assertEqual(f.types, correct)

        @vectorize
        def g(a, b):
            return a / b
        print(g(2.0, 3.0))
        print(g(2, 3))
        print(g.types)
        correct = ['dd->d']
        self.assertEqual(g.types, correct)