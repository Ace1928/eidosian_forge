import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
class AssertFalseTests(unittest.SynchronousTestCase):
    """
    Tests for L{SynchronousTestCase}'s C{assertFalse} and C{failIf} assertion
    methods.

    This is pretty paranoid.  Still, a certain paranoia is healthy if you
    are testing a unit testing framework.

    @note: As of 11.2, C{assertFalse} is preferred over C{failIf}.
    """

    def _assertFalseFalse(self, method):
        """
        Perform the positive case test for C{failIf} or C{assertFalse}.

        @param method: The test method to test.
        """
        for notTrue in [0, 0.0, False, None, (), []]:
            result = method(notTrue, f'failed on {notTrue!r}')
            if result != notTrue:
                self.fail(f'Did not return argument {notTrue!r}')

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

    def test_failIfFalse(self):
        """
        L{SynchronousTestCase.failIf} returns its argument if its argument is
        not considered true.
        """
        self._assertFalseFalse(self.failIf)

    def test_assertFalseFalse(self):
        """
        L{SynchronousTestCase.assertFalse} returns its argument if its argument
        is not considered true.
        """
        self._assertFalseFalse(self.assertFalse)

    def test_failIfTrue(self):
        """
        L{SynchronousTestCase.failIf} raises
        L{SynchronousTestCase.failureException} if its argument is considered
        true.
        """
        self._assertFalseTrue(self.failIf)

    def test_assertFalseTrue(self):
        """
        L{SynchronousTestCase.assertFalse} raises
        L{SynchronousTestCase.failureException} if its argument is considered
        true.
        """
        self._assertFalseTrue(self.assertFalse)