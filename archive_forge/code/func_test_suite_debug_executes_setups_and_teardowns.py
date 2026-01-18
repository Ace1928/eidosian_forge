import io
import sys
import unittest
def test_suite_debug_executes_setups_and_teardowns(self):
    ordering = []

    class Module(object):

        @staticmethod
        def setUpModule():
            ordering.append('setUpModule')

        @staticmethod
        def tearDownModule():
            ordering.append('tearDownModule')

    class Test(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            ordering.append('setUpClass')

        @classmethod
        def tearDownClass(cls):
            ordering.append('tearDownClass')

        def test_something(self):
            ordering.append('test_something')
    Test.__module__ = 'Module'
    sys.modules['Module'] = Module
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Test)
    suite.debug()
    expectedOrder = ['setUpModule', 'setUpClass', 'test_something', 'tearDownClass', 'tearDownModule']
    self.assertEqual(ordering, expectedOrder)