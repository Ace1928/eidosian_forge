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
def assertSamePath(self, first, second):
    """
        Assert that the two paths are the same, considering case normalization
        appropriate for the current platform.

        @type first: L{FilePath}
        @type second: L{FilePath}

        @raise C{self.failureType}: If the paths are not the same.
        """
    self.assertTrue(normcase(first.path) == normcase(second.path), f'{first!r} != {second!r}')