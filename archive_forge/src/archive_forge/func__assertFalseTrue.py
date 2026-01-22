import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def _assertFalseTrue(self, method):
    """
        Perform the negative case test for C{failIf} or C{assertFalse}.

        @param method: The test method to test.
        """
    for true in [1, True, 'cat', [1, 2], (3, 4)]:
        try:
            method(true, f'failed on {true!r}')
        except self.failureException as e:
            self.assertIn(f'failed on {true!r}', str(e), f'Raised incorrect exception on {true!r}: {e!r}')
        else:
            self.fail("Call to %s(%r) didn't fail" % (method.__name__, true))