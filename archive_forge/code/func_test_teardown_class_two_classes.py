import io
import sys
import unittest
def test_teardown_class_two_classes(self):

    class Test(unittest.TestCase):
        tearDownCalled = 0

        @classmethod
        def tearDownClass(cls):
            Test.tearDownCalled += 1
            unittest.TestCase.tearDownClass()

        def test_one(self):
            pass

        def test_two(self):
            pass

    class Test2(unittest.TestCase):
        tearDownCalled = 0

        @classmethod
        def tearDownClass(cls):
            Test2.tearDownCalled += 1
            unittest.TestCase.tearDownClass()

        def test_one(self):
            pass

        def test_two(self):
            pass
    result = self.runTests(Test, Test2)
    self.assertEqual(Test.tearDownCalled, 1)
    self.assertEqual(Test2.tearDownCalled, 1)
    self.assertEqual(result.testsRun, 4)
    self.assertEqual(len(result.errors), 0)