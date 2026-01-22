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
class DeprecatedAttributeTests(SynchronousTestCase):
    """
    Tests for L{twisted.python.deprecate._DeprecatedAttribute} and
    L{twisted.python.deprecate.deprecatedModuleAttribute}, which issue
    warnings for deprecated module-level attributes.
    """

    def setUp(self):
        self.version = deprecatedattributes.version
        self.message = deprecatedattributes.message
        self._testModuleName = __name__ + '.foo'

    def _getWarningString(self, attr):
        """
        Create the warning string used by deprecated attributes.
        """
        return _getDeprecationWarningString(deprecatedattributes.__name__ + '.' + attr, deprecatedattributes.version, DEPRECATION_WARNING_FORMAT + ': ' + deprecatedattributes.message)

    def test_deprecatedAttributeHelper(self):
        """
        L{twisted.python.deprecate._DeprecatedAttribute} correctly sets its
        __name__ to match that of the deprecated attribute and emits a warning
        when the original attribute value is accessed.
        """
        name = 'ANOTHER_DEPRECATED_ATTRIBUTE'
        setattr(deprecatedattributes, name, 42)
        attr = deprecate._DeprecatedAttribute(deprecatedattributes, name, self.version, self.message)
        self.assertEqual(attr.__name__, name)

        def addStackLevel():
            attr.get()
        addStackLevel()
        warningsShown = self.flushWarnings([self.test_deprecatedAttributeHelper])
        self.assertIs(warningsShown[0]['category'], DeprecationWarning)
        self.assertEqual(warningsShown[0]['message'], self._getWarningString(name))
        self.assertEqual(len(warningsShown), 1)

    def test_deprecatedAttribute(self):
        """
        L{twisted.python.deprecate.deprecatedModuleAttribute} wraps a
        module-level attribute in an object that emits a deprecation warning
        when it is accessed the first time only, while leaving other unrelated
        attributes alone.
        """
        deprecatedattributes.ANOTHER_ATTRIBUTE
        warningsShown = self.flushWarnings([self.test_deprecatedAttribute])
        self.assertEqual(len(warningsShown), 0)
        name = 'DEPRECATED_ATTRIBUTE'
        getattr(deprecatedattributes, name)
        warningsShown = self.flushWarnings([self.test_deprecatedAttribute])
        self.assertEqual(len(warningsShown), 1)
        self.assertIs(warningsShown[0]['category'], DeprecationWarning)
        self.assertEqual(warningsShown[0]['message'], self._getWarningString(name))

    def test_wrappedModule(self):
        """
        Deprecating an attribute in a module replaces and wraps that module
        instance, in C{sys.modules}, with a
        L{twisted.python.deprecate._ModuleProxy} instance but only if it hasn't
        already been wrapped.
        """
        sys.modules[self._testModuleName] = mod = types.ModuleType('foo')
        self.addCleanup(sys.modules.pop, self._testModuleName)
        setattr(mod, 'first', 1)
        setattr(mod, 'second', 2)
        deprecate.deprecatedModuleAttribute(Version('Twisted', 8, 0, 0), 'message', self._testModuleName, 'first')
        proxy = sys.modules[self._testModuleName]
        self.assertNotEqual(proxy, mod)
        deprecate.deprecatedModuleAttribute(Version('Twisted', 8, 0, 0), 'message', self._testModuleName, 'second')
        self.assertIs(proxy, sys.modules[self._testModuleName])