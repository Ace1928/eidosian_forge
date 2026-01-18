import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_requireModuleImportError(self):
    """
        When module import fails with ImportError it returns the specified
        default value.
        """
    for name in ['nosuchmtopodule', 'no.such.module']:
        default = object()
        result = reflect.requireModule(name, default=default)
        self.assertIs(result, default)