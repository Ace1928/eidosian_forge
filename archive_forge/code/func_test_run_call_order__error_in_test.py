import unittest
from unittest.test.support import LoggingResult
def test_run_call_order__error_in_test(self):
    events = []
    result = LoggingResult(events)

    def setUp():
        events.append('setUp')

    def test():
        events.append('test')
        raise RuntimeError('raised by test')

    def tearDown():
        events.append('tearDown')
    expected = ['startTest', 'setUp', 'test', 'addError', 'tearDown', 'stopTest']
    unittest.FunctionTestCase(test, setUp, tearDown).run(result)
    self.assertEqual(events, expected)