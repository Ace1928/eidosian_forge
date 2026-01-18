import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_requireModuleDefaultNone(self):
    """
        When module import fails it returns L{None} by default.
        """
    result = reflect.requireModule('no.such.module')
    self.assertIsNone(result)