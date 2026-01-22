import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def _assertTrueFalse(self, method):
    """
        Perform the negative case test for C{assertTrue} and C{failUnless}.

        @param method: The test method to test.
        """
    for notTrue in [0, 0.0, False, None, (), []]:
        try:
            method(notTrue, f'failed on {notTrue!r}')
        except self.failureException as e:
            self.assertIn(f'failed on {notTrue!r}', str(e), f'Raised incorrect exception on {notTrue!r}: {e!r}')
        else:
            self.fail("Call to %s(%r) didn't fail" % (method.__name__, notTrue))