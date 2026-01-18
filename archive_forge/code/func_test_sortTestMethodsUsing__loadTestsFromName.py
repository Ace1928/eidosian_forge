import functools
import sys
import types
import warnings
import unittest
def test_sortTestMethodsUsing__loadTestsFromName(self):

    def reversed_cmp(x, y):
        return -((x > y) - (x < y))
    m = types.ModuleType('m')

    class Foo(unittest.TestCase):

        def test_1(self):
            pass

        def test_2(self):
            pass
    m.Foo = Foo
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = reversed_cmp
    tests = loader.suiteClass([Foo('test_2'), Foo('test_1')])
    self.assertEqual(loader.loadTestsFromName('Foo', m), tests)