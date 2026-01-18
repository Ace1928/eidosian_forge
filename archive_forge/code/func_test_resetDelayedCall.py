import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
def test_resetDelayedCall(self):
    """
        If a delayed call is reset, the timeout passed to C{doIteration} is
        based on the interval between the time when reset is called and the
        new delay of the call.
        """
    reactor = TimeoutReportReactor()
    call = reactor.callLater(50, lambda: None)
    reactor.now += 25
    call.reset(15)
    timeout = self._checkIterationTimeout(reactor)
    self.assertEqual(timeout, 15)