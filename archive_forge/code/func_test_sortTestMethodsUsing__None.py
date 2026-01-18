import functools
import sys
import types
import warnings
import unittest
def test_sortTestMethodsUsing__None(self):

    class Foo(unittest.TestCase):

        def test_1(self):
            pass

        def test_2(self):
            pass
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None
    test_names = ['test_2', 'test_1']
    self.assertEqual(set(loader.getTestCaseNames(Foo)), set(test_names))