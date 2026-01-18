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
def test_passed_unspecifiedKeyword(self):
    """
        L{passed} raises a L{TypeError} if a keyword argument not
        present in the function's declaration is passed.
        """

    def func(a):
        pass
    self.assertRaises(TypeError, self.checkPassed, func, 1, z=2)