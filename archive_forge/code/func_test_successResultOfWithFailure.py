import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_successResultOfWithFailure(self):
    """
        L{SynchronousTestCase.successResultOf} raises
        L{SynchronousTestCase.failureException} when called with a L{Deferred}
        with a failure result.
        """
    self.assertRaises(self.failureException, self.successResultOf, fail(self.failure))