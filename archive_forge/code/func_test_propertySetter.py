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