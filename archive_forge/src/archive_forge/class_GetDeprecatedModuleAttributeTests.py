import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
class GetDeprecatedModuleAttributeTests(unittest.SynchronousTestCase):
    """
    Test L{SynchronousTestCase.getDeprecatedModuleAttribute}

    @ivar version: The version at which L{test_assertions.somethingOld}
        is marked deprecated.
    @type version: L{incremental.Version}
    """
    version = Version('Bar', 1, 2, 3)

    def test_deprecated(self):
        """
        L{getDeprecatedModuleAttribute} returns the specified attribute and
        consumes the deprecation warning that generates.
        """
        self.assertIs(_somethingOld, self.getDeprecatedModuleAttribute(__name__, 'somethingOld', self.version))
        self.assertEqual([], self.flushWarnings())

    def test_message(self):
        """
        The I{message} argument to L{getDeprecatedModuleAttribute} matches the
        prefix of the deprecation message.
        """
        self.assertIs(_somethingOld, self.getDeprecatedModuleAttribute(__name__, 'somethingOld', self.version, message="It's old"))
        self.assertEqual([], self.flushWarnings())

    def test_messageMismatch(self):
        """
        L{getDeprecatedModuleAttribute} fails the test if the I{message} isn't
        part of the deprecation message prefix.
        """
        self.assertRaises(self.failureException, self.getDeprecatedModuleAttribute, __name__, 'somethingOld', self.version, "It's shiny and new")
        self.assertEqual([], self.flushWarnings())

    def test_notDeprecated(self):
        """
        L{getDeprecatedModuleAttribute} fails the test when used to get an
        attribute that isn't actually deprecated.
        """
        self.assertRaises(self.failureException, self.getDeprecatedModuleAttribute, __name__, 'somethingNew', self.version)