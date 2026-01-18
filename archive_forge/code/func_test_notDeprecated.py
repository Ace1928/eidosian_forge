import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_notDeprecated(self):
    """
        L{getDeprecatedModuleAttribute} fails the test when used to get an
        attribute that isn't actually deprecated.
        """
    self.assertRaises(self.failureException, self.getDeprecatedModuleAttribute, __name__, 'somethingNew', self.version)