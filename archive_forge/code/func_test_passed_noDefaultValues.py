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
def test_passed_noDefaultValues(self):
    """
        The results of L{passed} only include arguments explicitly
        passed, not default values.
        """

    def func(a, b, c=1, d=2, e=3):
        pass
    self.assertEqual(self.checkPassed(func, 1, 2, e=7), dict(a=1, b=2, e=7))