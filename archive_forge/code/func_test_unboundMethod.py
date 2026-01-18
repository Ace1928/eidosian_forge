import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_unboundMethod(self):
    """
        L{fullyQualifiedName} returns the name of an unbound method inside its
        class and its module.
        """
    self._checkFullyQualifiedName(self.__class__.test_unboundMethod, f'{__name__}.{self.__class__.__name__}.test_unboundMethod')