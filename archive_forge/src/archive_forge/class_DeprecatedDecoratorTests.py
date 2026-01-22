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
class DeprecatedDecoratorTests(SynchronousTestCase):
    """
    Tests for deprecated decorators.
    """

    def assertDocstring(self, target, expected):
        """
        Check that C{target} object has the C{expected} docstring lines.

        @param target: Object which is checked.
        @type target: C{anything}

        @param expected: List of lines, ignoring empty lines or leading or
            trailing spaces.
        @type expected: L{list} or L{str}
        """
        self.assertEqual(expected, [x.strip() for x in target.__doc__.splitlines() if x.strip()])

    def test_propertyGetter(self):
        """
        When L{deprecatedProperty} is used on a C{property}, accesses raise a
        L{DeprecationWarning} and getter docstring is updated to inform the
        version in which it was deprecated. C{deprecatedVersion} attribute is
        also set to inform the deprecation version.
        """
        obj = ClassWithDeprecatedProperty()
        obj.someProperty
        self.assertDocstring(ClassWithDeprecatedProperty.someProperty, ['Getter docstring.', '@return: The property.', 'Deprecated in Twisted 1.2.3.'])
        ClassWithDeprecatedProperty.someProperty.deprecatedVersion = Version('Twisted', 1, 2, 3)
        message = 'twisted.python.test.test_deprecate.ClassWithDeprecatedProperty.someProperty was deprecated in Twisted 1.2.3'
        warnings = self.flushWarnings([self.test_propertyGetter])
        self.assertEqual(1, len(warnings))
        self.assertEqual(DeprecationWarning, warnings[0]['category'])
        self.assertEqual(message, warnings[0]['message'])

    def test_propertySetter(self):
        """
        When L{deprecatedProperty} is used on a C{property}, setter accesses
        raise a L{DeprecationWarning}.
        """
        newValue = object()
        obj = ClassWithDeprecatedProperty()
        obj.someProperty = newValue
        self.assertIs(newValue, obj._someProtectedValue)
        message = 'twisted.python.test.test_deprecate.ClassWithDeprecatedProperty.someProperty was deprecated in Twisted 1.2.3'
        warnings = self.flushWarnings([self.test_propertySetter])
        self.assertEqual(1, len(warnings))
        self.assertEqual(DeprecationWarning, warnings[0]['category'])
        self.assertEqual(message, warnings[0]['message'])

    def test_class(self):
        """
        When L{deprecated} is used on a class, instantiations raise a
        L{DeprecationWarning} and class's docstring is updated to inform the
        version in which it was deprecated. C{deprecatedVersion} attribute is
        also set to inform the deprecation version.
        """
        DeprecatedClass()
        self.assertDocstring(DeprecatedClass, ['Class which is entirely deprecated without having a replacement.', 'Deprecated in Twisted 1.2.3.'])
        DeprecatedClass.deprecatedVersion = Version('Twisted', 1, 2, 3)
        message = 'twisted.python.test.test_deprecate.DeprecatedClass was deprecated in Twisted 1.2.3'
        warnings = self.flushWarnings([self.test_class])
        self.assertEqual(1, len(warnings))
        self.assertEqual(DeprecationWarning, warnings[0]['category'])
        self.assertEqual(message, warnings[0]['message'])

    def test_deprecatedReplacement(self):
        """
        L{deprecated} takes an additional replacement parameter that can be used
        to indicate the new, non-deprecated method developers should use.  If
        the replacement parameter is a string, it will be interpolated directly
        into the warning message.
        """
        version = Version('Twisted', 8, 0, 0)
        dummy = deprecated(version, 'something.foobar')(dummyCallable)
        self.assertEqual(dummy.__doc__, '\n    Do nothing.\n\n    This is used to test the deprecation decorators.\n\n    Deprecated in Twisted 8.0.0; please use something.foobar instead.\n    ')

    def test_deprecatedReplacementWithCallable(self):
        """
        L{deprecated} takes an additional replacement parameter that can be used
        to indicate the new, non-deprecated method developers should use.  If
        the replacement parameter is a callable, its fully qualified name will
        be interpolated into the warning message.
        """
        version = Version('Twisted', 8, 0, 0)
        decorator = deprecated(version, replacement=dummyReplacementMethod)
        dummy = decorator(dummyCallable)
        self.assertEqual(dummy.__doc__, '\n    Do nothing.\n\n    This is used to test the deprecation decorators.\n\n    Deprecated in Twisted 8.0.0; please use %s.dummyReplacementMethod instead.\n    ' % (__name__,))

    def test_deprecatedKeywordParameter(self):
        message = "The 'foo' parameter to twisted.python.test.test_deprecate.functionWithDeprecatedParameter was deprecated in Twisted 19.2.0"
        with catch_warnings(record=True) as ws:
            simplefilter('always')
            functionWithDeprecatedParameter(10, 20)
            self.assertEqual(ws, [])
            functionWithDeprecatedParameter(10, 20, 30)
            self.assertEqual(ws, [])
            functionWithDeprecatedParameter(10, 20, foo=40)
            self.assertEqual(len(ws), 1)
            self.assertEqual(ws[0].category, DeprecationWarning)
            self.assertEqual(str(ws[0].message), message)
            ws.clear()
            functionWithDeprecatedParameter(10, 20, bar=50)
            self.assertEqual(ws, [])
            functionWithDeprecatedParameter(10, 20, 30, 40)
            self.assertEqual(len(ws), 1)
            self.assertEqual(ws[0].category, DeprecationWarning)
            self.assertEqual(str(ws[0].message), message)