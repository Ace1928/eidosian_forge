import inspect
import sys
import types
import warnings
from os.path import normcase
from warnings import catch_warnings, simplefilter
from incremental import Version
from twisted.python import deprecate
from twisted.python.deprecate import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.python.test import deprecatedattributes
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SynchronousTestCase
from twisted.python.deprecate import deprecatedModuleAttribute
from incremental import Version
from twisted.python import deprecate
from twisted.python import deprecate
class DeprecationWarningsTests(SynchronousTestCase):

    def test_getDeprecationWarningString(self):
        """
        L{getDeprecationWarningString} returns a string that tells us that a
        callable was deprecated at a certain released version of Twisted.
        """
        version = Version('Twisted', 8, 0, 0)
        self.assertEqual(getDeprecationWarningString(self.test_getDeprecationWarningString, version), '%s.DeprecationWarningsTests.test_getDeprecationWarningString was deprecated in Twisted 8.0.0' % (__name__,))

    def test_getDeprecationWarningStringWithFormat(self):
        """
        L{getDeprecationWarningString} returns a string that tells us that a
        callable was deprecated at a certain released version of Twisted, with
        a message containing additional information about the deprecation.
        """
        version = Version('Twisted', 8, 0, 0)
        format = DEPRECATION_WARNING_FORMAT + ': This is a message'
        self.assertEqual(getDeprecationWarningString(self.test_getDeprecationWarningString, version, format), '%s.DeprecationWarningsTests.test_getDeprecationWarningString was deprecated in Twisted 8.0.0: This is a message' % (__name__,))

    def test_deprecateEmitsWarning(self):
        """
        Decorating a callable with L{deprecated} emits a warning.
        """
        version = Version('Twisted', 8, 0, 0)
        dummy = deprecated(version)(dummyCallable)

        def addStackLevel():
            dummy()
        with catch_warnings(record=True) as caught:
            simplefilter('always')
            addStackLevel()
            self.assertEqual(caught[0].category, DeprecationWarning)
            self.assertEqual(str(caught[0].message), getDeprecationWarningString(dummyCallable, version))
            self.assertEqual(caught[0].filename.rstrip('co'), __file__.rstrip('co'))

    def test_deprecatedPreservesName(self):
        """
        The decorated function has the same name as the original.
        """
        version = Version('Twisted', 8, 0, 0)
        dummy = deprecated(version)(dummyCallable)
        self.assertEqual(dummyCallable.__name__, dummy.__name__)
        self.assertEqual(fullyQualifiedName(dummyCallable), fullyQualifiedName(dummy))

    def test_getDeprecationDocstring(self):
        """
        L{_getDeprecationDocstring} returns a note about the deprecation to go
        into a docstring.
        """
        version = Version('Twisted', 8, 0, 0)
        self.assertEqual('Deprecated in Twisted 8.0.0.', _getDeprecationDocstring(version, ''))

    def test_deprecatedUpdatesDocstring(self):
        """
        The docstring of the deprecated function is appended with information
        about the deprecation.
        """

        def localDummyCallable():
            """
            Do nothing.

            This is used to test the deprecation decorators.
            """
        version = Version('Twisted', 8, 0, 0)
        dummy = deprecated(version)(localDummyCallable)
        _appendToDocstring(localDummyCallable, _getDeprecationDocstring(version, ''))
        self.assertEqual(localDummyCallable.__doc__, dummy.__doc__)

    def test_versionMetadata(self):
        """
        Deprecating a function adds version information to the decorated
        version of that function.
        """
        version = Version('Twisted', 8, 0, 0)
        dummy = deprecated(version)(dummyCallable)
        self.assertEqual(version, dummy.deprecatedVersion)

    def test_getDeprecationWarningStringReplacement(self):
        """
        L{getDeprecationWarningString} takes an additional replacement parameter
        that can be used to add information to the deprecation.  If the
        replacement parameter is a string, it will be interpolated directly into
        the result.
        """
        version = Version('Twisted', 8, 0, 0)
        warningString = getDeprecationWarningString(self.test_getDeprecationWarningString, version, replacement='something.foobar')
        self.assertEqual(warningString, '%s was deprecated in Twisted 8.0.0; please use something.foobar instead' % (fullyQualifiedName(self.test_getDeprecationWarningString),))

    def test_getDeprecationWarningStringReplacementWithCallable(self):
        """
        L{getDeprecationWarningString} takes an additional replacement parameter
        that can be used to add information to the deprecation. If the
        replacement parameter is a callable, its fully qualified name will be
        interpolated into the result.
        """
        version = Version('Twisted', 8, 0, 0)
        warningString = getDeprecationWarningString(self.test_getDeprecationWarningString, version, replacement=dummyReplacementMethod)
        self.assertEqual(warningString, '%s was deprecated in Twisted 8.0.0; please use %s.dummyReplacementMethod instead' % (fullyQualifiedName(self.test_getDeprecationWarningString), __name__))