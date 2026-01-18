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
def test_tcpNoDelay(self):
    """
        The transport of a protocol connected with L{IReactorTCP.connectTCP} or
        L{IReactor.TCP.listenTCP} can have its I{TCP_NODELAY} state inspected
        and manipulated with L{ITCPTransport.getTcpNoDelay} and
        L{ITCPTransport.setTcpNoDelay}.
        """

    def check(serverProtocol, clientProtocol):
        for p in [serverProtocol, clientProtocol]:
            transport = p.transport
            self.assertEqual(transport.getTcpNoDelay(), 0)
            transport.setTcpNoDelay(1)
            self.assertEqual(transport.getTcpNoDelay(), 1)
            transport.setTcpNoDelay(0)
            self.assertEqual(transport.getTcpNoDelay(), 0)
    return self._connectedClientAndServerTest(check)