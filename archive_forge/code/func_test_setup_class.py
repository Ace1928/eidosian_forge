import io
import sys
import unittest
def test_setup_class(self):

    class Test(unittest.TestCase):
        setUpCalled = 0

        @classmethod
        def setUpClass(cls):
            Test.setUpCalled += 1
            unittest.TestCase.setUpClass()

        def test_one(self):
            pass

        def test_two(self):
            pass
    result = self.runTests(Test)
    self.assertEqual(Test.setUpCalled, 1)
    self.assertEqual(result.testsRun, 2)
    self.assertEqual(len(result.errors), 0)