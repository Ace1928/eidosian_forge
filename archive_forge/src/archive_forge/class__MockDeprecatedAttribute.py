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
class _MockDeprecatedAttribute:
    """
    Mock of L{twisted.python.deprecate._DeprecatedAttribute}.

    @ivar value: The value of the attribute.
    """

    def __init__(self, value):
        self.value = value

    def get(self):
        """
        Get a known value.
        """
        return self.value