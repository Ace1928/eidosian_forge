import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_run__requires_result(self):
    suite = unittest.TestSuite()
    try:
        suite.run()
    except TypeError:
        pass
    else:
        self.fail('Failed to raise TypeError')