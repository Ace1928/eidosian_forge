import functools
import sys
import types
import warnings
import unittest
def test_testMethodPrefix__loadTestsFromModule(self):
    m = types.ModuleType('m')

    class Foo(unittest.TestCase):

        def test_1(self):
            pass

        def test_2(self):
            pass

        def foo_bar(self):
            pass
    m.Foo = Foo
    tests_1 = [unittest.TestSuite([Foo('foo_bar')])]
    tests_2 = [unittest.TestSuite([Foo('test_1'), Foo('test_2')])]
    loader = unittest.TestLoader()
    loader.testMethodPrefix = 'foo'
    self.assertEqual(list(loader.loadTestsFromModule(m)), tests_1)
    loader.testMethodPrefix = 'test'
    self.assertEqual(list(loader.loadTestsFromModule(m)), tests_2)