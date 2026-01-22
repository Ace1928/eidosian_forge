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
def _completeWith(self, completionState: SchedulerError, deferredResult: Union[Iterator[_TaskResultT], Failure]) -> None:
    """
        @param completionState: a L{SchedulerError} exception or a subclass
            thereof, indicating what exception should be raised when subsequent
            operations are performed.

        @param deferredResult: the result to fire all the deferreds with.
        """
    self._completionState = completionState
    self._completionResult = deferredResult
    if not self._pauseCount:
        self._cooperator._removeTask(self)
    for d in self._deferreds:
        d.callback(deferredResult)