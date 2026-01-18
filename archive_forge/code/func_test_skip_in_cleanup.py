import unittest
from unittest.test.support import LoggingResult
def test_skip_in_cleanup(self):

    class Foo(unittest.TestCase):

        def test_skip_me(self):
            pass

        def tearDown(self):
            self.skipTest('skip')
    events = []
    result = LoggingResult(events)
    test = Foo('test_skip_me')
    self.assertIs(test.run(result), result)
    self.assertEqual(events, ['startTest', 'addSkip', 'stopTest'])
    self.assertEqual(result.skipped, [(test, 'skip')])