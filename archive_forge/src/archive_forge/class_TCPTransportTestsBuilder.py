import errno
import gc
import io
import os
import socket
from functools import wraps
from typing import Callable, ClassVar, List, Mapping, Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass, verifyObject
import attr
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import (
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import (
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.test.test_tcp import (
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
class TCPTransportTestsBuilder(TCPTransportServerAddressTestMixin, WriteSequenceTestsMixin, ReactorBuilder):
    """
    Test standard L{ITCPTransport}s built with C{listenTCP} and C{connectTCP}.
    """

    def getConnectedClientAndServer(self, reactor, interface, addressFamily):
        """
        Return a L{Deferred} firing with a L{MyClientFactory} and
        L{MyServerFactory} connected pair, and the listening C{Port}.
        """
        server = MyServerFactory()
        server.protocolConnectionMade = Deferred()
        server.protocolConnectionLost = Deferred()
        client = MyClientFactory()
        client.protocolConnectionMade = Deferred()
        client.protocolConnectionLost = Deferred()
        port = reactor.listenTCP(0, server, interface=interface)
        lostDeferred = gatherResults([client.protocolConnectionLost, server.protocolConnectionLost])

        def stop(result):
            reactor.stop()
            return result
        lostDeferred.addBoth(stop)
        startDeferred = gatherResults([client.protocolConnectionMade, server.protocolConnectionMade])
        deferred = Deferred()

        def start(protocols):
            client, server = protocols
            log.msg('client connected %s' % client)
            log.msg('server connected %s' % server)
            deferred.callback((client, server, port))
        startDeferred.addCallback(start)
        reactor.connectTCP(interface, port.getHost().port, client)
        return deferred