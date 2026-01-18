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
def test_stopStartReading(self):
    """
        This test verifies transport socket read state after multiple
        pause/resumeProducing calls.
        """
    sf = ServerFactory()
    reactor = sf.reactor = self.buildReactor()
    skippedReactors = ['Glib2Reactor', 'Gtk2Reactor']
    reactorClassName = reactor.__class__.__name__
    if reactorClassName in skippedReactors and platform.isWindows():
        raise SkipTest('This test is broken on gtk/glib under Windows.')
    sf.protocol = StopStartReadingProtocol
    sf.ready = Deferred()
    sf.stop = Deferred()
    p = reactor.listenTCP(0, sf)
    port = p.getHost().port

    def proceed(protos, port):
        """
            Send several IOCPReactor's buffers' worth of data.
            """
        self.assertTrue(protos[0])
        self.assertTrue(protos[1])
        protos = (protos[0][1], protos[1][1])
        protos[0].transport.write(b'x' * (2 * 4096) + b'y' * (2 * 4096))
        return sf.stop.addCallback(cleanup, protos, port).addCallback(lambda ign: reactor.stop())

    def cleanup(data, protos, port):
        """
            Make sure IOCPReactor didn't start several WSARecv operations
            that clobbered each other's results.
            """
        self.assertEqual(data, b'x' * (2 * 4096) + b'y' * (2 * 4096), 'did not get the right data')
        return DeferredList([maybeDeferred(protos[0].transport.loseConnection), maybeDeferred(protos[1].transport.loseConnection), maybeDeferred(port.stopListening)])
    cc = TCP4ClientEndpoint(reactor, '127.0.0.1', port)
    cf = ClientFactory()
    cf.protocol = Protocol
    d = DeferredList([cc.connect(cf), sf.ready]).addCallback(proceed, p)
    d.addErrback(log.err)
    self.runReactor(reactor)