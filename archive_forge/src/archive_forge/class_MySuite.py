import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
class MySuite(unittest.TestSuite):
    called = False

    def __call__(self, *args, **kw):
        self.called = True
        unittest.TestSuite.__call__(self, *args, **kw)