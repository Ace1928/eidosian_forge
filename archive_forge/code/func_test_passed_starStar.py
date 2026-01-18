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
def test_passed_starStar(self):
    """
        Additional keyword arguments are passed as a dict to the star star
        keyword argument.
        """

    def func(a, **b):
        pass
    self.assertEqual(self.checkPassed(func, 1, x=2, y=3, z=4), dict(a=1, b=dict(x=2, y=3, z=4)))