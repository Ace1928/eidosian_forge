import inspect
import warnings
from typing import Callable, List
from zope.interface import implementer
from typing_extensions import ParamSpec
from twisted.internet import defer, utils
from twisted.python import failure
from twisted.trial import itrial, util
from twisted.trial._synctest import FailTest, SkipTest, SynchronousTestCase
def _undeprecateReactor(self, reactor):
    """
        Restore the deprecated reactor methods. Undoes what
        L{_deprecateReactor} did.

        @param reactor: The Twisted reactor.
        """
    for name, method in self._reactorMethods.items():
        setattr(reactor, name, method)
    self._reactorMethods = {}