import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_requireModuleRequestedImport(self):
    """
        When module import succeed it returns the module and not the default
        value.
        """
    from twisted.python import monkey
    default = object()
    self.assertIs(reflect.requireModule('twisted.python.monkey', default=default), monkey)