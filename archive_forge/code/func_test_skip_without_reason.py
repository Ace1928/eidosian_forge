import unittest
from unittest.test.support import LoggingResult
def test_skip_without_reason(self):

    class Foo(unittest.TestCase):

        @unittest.skip
        def test_1(self):
            pass
    result = unittest.TestResult()
    test = Foo('test_1')
    suite = unittest.TestSuite([test])
    self.assertIs(suite.run(result), result)
    self.assertEqual(result.skipped, [(test, '')])