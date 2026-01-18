import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_run__empty_suite(self):
    events = []
    result = LoggingResult(events)
    suite = unittest.TestSuite()
    suite.run(result)
    self.assertEqual(events, [])