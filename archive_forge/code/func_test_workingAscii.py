import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_workingAscii(self):
    """
        L{safe_str} for C{str} with ascii-only data should return the
        value unchanged.
        """
    x = 'a'
    self.assertEqual(reflect.safe_str(x), 'a')