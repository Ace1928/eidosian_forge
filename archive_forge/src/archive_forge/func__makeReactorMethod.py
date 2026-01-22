import inspect
import warnings
from typing import Callable, List
from zope.interface import implementer
from typing_extensions import ParamSpec
from twisted.internet import defer, utils
from twisted.python import failure
from twisted.trial import itrial, util
from twisted.trial._synctest import FailTest, SkipTest, SynchronousTestCase
def _makeReactorMethod(self, name):
    """
        Create a method which wraps the reactor method C{name}. The new
        method issues a deprecation warning and calls the original.
        """

    def _(*a, **kw):
        warnings.warn('reactor.%s cannot be used inside unit tests. In the future, using %s will fail the test and may crash or hang the test run.' % (name, name), stacklevel=2, category=DeprecationWarning)
        return self._reactorMethods[name](*a, **kw)
    return _