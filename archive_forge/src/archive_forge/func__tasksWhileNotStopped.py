import sys
import time
import warnings
from typing import (
from zope.interface import implementer
from incremental import Version
from twisted.internet.base import DelayedCall
from twisted.internet.defer import Deferred, ensureDeferred, maybeDeferred
from twisted.internet.error import ReactorNotRunning
from twisted.internet.interfaces import IDelayedCall, IReactorCore, IReactorTime
from twisted.python import log, reflect
from twisted.python.deprecate import _getDeprecationWarningString
from twisted.python.failure import Failure
def _tasksWhileNotStopped(self) -> Iterable[CooperativeTask]:
    """
        Yield all L{CooperativeTask} objects in a loop as long as this
        L{Cooperator}'s termination condition has not been met.
        """
    terminator = self._terminationPredicateFactory()
    while self._tasks:
        for t in self._metarator:
            yield t
            if terminator():
                return
        self._metarator = iter(self._tasks)