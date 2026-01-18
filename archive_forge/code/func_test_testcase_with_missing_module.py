import io
import sys
import unittest
def test_testcase_with_missing_module(self):

    class Test(unittest.TestCase):

        def test_one(self):
            pass

        def test_two(self):
            pass
    Test.__module__ = 'Module'
    sys.modules.pop('Module', None)
    result = self.runTests(Test)
    self.assertEqual(result.testsRun, 2)