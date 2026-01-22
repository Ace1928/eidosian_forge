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
class StreamingProducerClient(ConnectableProtocol):
    """
    Call abortConnection() when the other side has stopped reading.

    In particular, we want to call abortConnection() only once our local
    socket hits a state where it is no longer writeable. This helps emulate
    the most common use case for abortConnection(), closing a connection after
    a timeout, with write buffers being full.

    Since it's very difficult to know when this actually happens, we just
    write a lot of data, and assume at that point no more writes will happen.
    """
    paused = False
    extraWrites = 0
    inReactorMethod = False

    def connectionMade(self):
        self.write()

    def write(self):
        """
        Write large amount to transport, then wait for a while for buffers to
        fill up.
        """
        self.transport.registerProducer(self, True)
        for i in range(100):
            self.transport.write(b'1234567890' * 32000)

    def resumeProducing(self):
        self.paused = False

    def stopProducing(self):
        pass

    def pauseProducing(self):
        """
        Called when local buffer fills up.

        The goal is to hit the point where the local file descriptor is not
        writeable (or the moral equivalent). The fact that pauseProducing has
        been called is not sufficient, since that can happen when Twisted's
        buffers fill up but OS hasn't gotten any writes yet. We want to be as
        close as possible to every buffer (including OS buffers) being full.

        So, we wait a bit more after this for Twisted to write out a few
        chunks, then abortConnection.
        """
        if self.paused:
            return
        self.paused = True
        self.reactor.callLater(0.01, self.doAbort)

    def doAbort(self):
        if not self.paused:
            log.err(RuntimeError('BUG: We should be paused a this point.'))
        self.inReactorMethod = True
        self.transport.abortConnection()
        self.inReactorMethod = False

    def connectionLost(self, reason):
        self.otherProtocol.transport.startReading()
        ConnectableProtocol.connectionLost(self, reason)