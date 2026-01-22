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
class MyProtocolFactoryMixin:
    """
    Mixin for factories which create L{AccumulatingProtocol} instances.

    @type protocolFactory: no-argument callable
    @ivar protocolFactory: Factory for protocols - takes the place of the
        typical C{protocol} attribute of factories (but that name is used by
        this class for something else).

    @type protocolConnectionMade: L{None} or L{defer.Deferred}
    @ivar protocolConnectionMade: When an instance of L{AccumulatingProtocol}
        is connected, if this is not L{None}, the L{Deferred} will be called
        back with the protocol instance and the attribute set to L{None}.

    @type protocolConnectionLost: L{None} or L{defer.Deferred}
    @ivar protocolConnectionLost: When an instance of L{AccumulatingProtocol}
        is created, this will be set as its C{closedDeferred} attribute and
        then this attribute will be set to L{None} so the L{defer.Deferred} is
        not used by more than one protocol.

    @ivar protocol: The most recently created L{AccumulatingProtocol} instance
        which was returned from C{buildProtocol}.

    @type called: C{int}
    @ivar called: A counter which is incremented each time C{buildProtocol}
        is called.

    @ivar peerAddresses: A C{list} of the addresses passed to C{buildProtocol}.
    """
    protocolFactory = AccumulatingProtocol
    protocolConnectionMade = None
    protocolConnectionLost = None
    protocol: Optional[Callable[[], Protocol]] = None
    called = 0

    def __init__(self):
        self.peerAddresses = []

    def buildProtocol(self, addr):
        """
        Create a L{AccumulatingProtocol} and set it up to be able to perform
        callbacks.
        """
        self.peerAddresses.append(addr)
        self.called += 1
        p = self.protocolFactory()
        p.factory = self
        p.closedDeferred = self.protocolConnectionLost
        self.protocolConnectionLost = None
        self.protocol = p
        return p