import unittest
from unittest.test.support import LoggingResult
def test_skipping_decorators(self):
    op_table = ((unittest.skipUnless, False, True), (unittest.skipIf, True, False))
    for deco, do_skip, dont_skip in op_table:

        class Foo(unittest.TestCase):

            def defaultTestResult(self):
                return LoggingResult(events)

            @deco(do_skip, 'testing')
            def test_skip(self):
                pass

            @deco(dont_skip, 'testing')
            def test_dont_skip(self):
                pass
        test_do_skip = Foo('test_skip')
        test_dont_skip = Foo('test_dont_skip')
        suite = unittest.TestSuite([test_do_skip, test_dont_skip])
        events = []
        result = LoggingResult(events)
        self.assertIs(suite.run(result), result)
        self.assertEqual(len(result.skipped), 1)
        expected = ['startTest', 'addSkip', 'stopTest', 'startTest', 'addSuccess', 'stopTest']
        self.assertEqual(events, expected)
        self.assertEqual(result.testsRun, 2)
        self.assertEqual(result.skipped, [(test_do_skip, 'testing')])
        self.assertTrue(result.wasSuccessful())
        events = []
        result = test_do_skip.run()
        self.assertEqual(events, ['startTestRun', 'startTest', 'addSkip', 'stopTest', 'stopTestRun'])
        self.assertEqual(result.skipped, [(test_do_skip, 'testing')])
        events = []
        result = test_dont_skip.run()
        self.assertEqual(events, ['startTestRun', 'startTest', 'addSuccess', 'stopTest', 'stopTestRun'])
        self.assertEqual(result.skipped, [])