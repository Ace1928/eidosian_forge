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
@classmethod
def withCount(cls, countCallable: Callable[[int], object]) -> 'LoopingCall':
    """
        An alternate constructor for L{LoopingCall} that makes available the
        number of calls which should have occurred since it was last invoked.

        Note that this number is an C{int} value; It represents the discrete
        number of calls that should have been made.  For example, if you are
        using a looping call to display an animation with discrete frames, this
        number would be the number of frames to advance.

        The count is normally 1, but can be higher. For example, if the reactor
        is blocked and takes too long to invoke the L{LoopingCall}, a Deferred
        returned from a previous call is not fired before an interval has
        elapsed, or if the callable itself blocks for longer than an interval,
        preventing I{itself} from being called.

        When running with an interval of 0, count will be always 1.

        @param countCallable: A callable that will be invoked each time the
            resulting LoopingCall is run, with an integer specifying the number
            of calls that should have been invoked.

        @return: An instance of L{LoopingCall} with call counting enabled,
            which provides the count as the first positional argument.

        @since: 9.0
        """

    def counter() -> object:
        now = self.clock.seconds()
        if self.interval == 0:
            self._realLastTime = now
            return countCallable(1)
        lastTime = self._realLastTime
        if lastTime is None:
            assert self.starttime is not None, 'LoopingCall called before it was started'
            lastTime = self.starttime
            if self._runAtStart:
                assert self.interval is not None, 'Looping call called with None interval'
                lastTime -= self.interval
        lastInterval = self._intervalOf(lastTime)
        thisInterval = self._intervalOf(now)
        count = thisInterval - lastInterval
        if count > 0:
            self._realLastTime = now
            return countCallable(count)
        return None
    self = cls(counter)
    return self