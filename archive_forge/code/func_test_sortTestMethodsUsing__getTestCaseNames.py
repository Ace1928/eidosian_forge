import functools
import sys
import types
import warnings
import unittest
def test_sortTestMethodsUsing__getTestCaseNames(self):

    def reversed_cmp(x, y):
        return -((x > y) - (x < y))

    class Foo(unittest.TestCase):

        def test_1(self):
            pass

        def test_2(self):
            pass
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = reversed_cmp
    test_names = ['test_2', 'test_1']
    self.assertEqual(loader.getTestCaseNames(Foo), test_names)