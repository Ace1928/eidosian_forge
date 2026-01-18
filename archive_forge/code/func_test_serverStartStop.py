import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
def test_serverStartStop(self):
    """
        The factory passed to L{IReactorTCP.listenTCP} should be started only
        when it transitions from being used on no ports to being used on one
        port and should be stopped only when it transitions from being used on
        one port to being used on no ports.
        """
    f = StartStopFactory()
    p1 = reactor.listenTCP(0, f, interface='127.0.0.1')
    self.addCleanup(p1.stopListening)
    self.assertEqual((f.started, f.stopped), (1, 0))
    p2 = reactor.listenTCP(0, f, interface='127.0.0.1')
    p3 = reactor.listenTCP(0, f, interface='127.0.0.1')
    self.assertEqual((f.started, f.stopped), (1, 0))
    d1 = defer.maybeDeferred(p1.stopListening)
    d2 = defer.maybeDeferred(p2.stopListening)
    closedDeferred = defer.gatherResults([d1, d2])

    def cbClosed(ignored):
        self.assertEqual((f.started, f.stopped), (1, 0))
        return p3.stopListening()
    closedDeferred.addCallback(cbClosed)

    def cbClosedAll(ignored):
        self.assertEqual((f.started, f.stopped), (1, 1))
    closedDeferred.addCallback(cbClosedAll)
    return closedDeferred