import functools
import sys
import types
import warnings
import unittest
def test_partial_functions(self):

    def noop(arg):
        pass

    class Foo(unittest.TestCase):
        pass
    setattr(Foo, 'test_partial', functools.partial(noop, None))
    loader = unittest.TestLoader()
    test_names = ['test_partial']
    self.assertEqual(loader.getTestCaseNames(Foo), test_names)