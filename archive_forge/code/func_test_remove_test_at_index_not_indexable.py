import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_remove_test_at_index_not_indexable(self):
    if not unittest.BaseTestSuite._cleanup:
        raise unittest.SkipTest('Suite cleanup is disabled')
    suite = unittest.TestSuite()
    suite._tests = None
    suite._removeTestAtIndex(2)