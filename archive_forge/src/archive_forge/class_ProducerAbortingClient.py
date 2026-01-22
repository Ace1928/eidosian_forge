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
class ProducerAbortingClient(ConnectableProtocol):
    """
    Call abortConnection from doWrite, via resumeProducing.
    """
    inReactorMethod = True
    producerStopped = False

    def write(self):
        self.transport.write(b'lalala' * 127000)
        self.inRegisterProducer = True
        self.transport.registerProducer(self, False)
        self.inRegisterProducer = False

    def connectionMade(self):
        self.write()

    def resumeProducing(self):
        self.inReactorMethod = True
        if not self.inRegisterProducer:
            self.transport.abortConnection()
        self.inReactorMethod = False

    def stopProducing(self):
        self.producerStopped = True

    def connectionLost(self, reason):
        if not self.producerStopped:
            raise RuntimeError('BUG: stopProducing() was never called.')
        if self.inReactorMethod:
            raise RuntimeError('BUG: connectionLost called re-entrantly!')
        ConnectableProtocol.connectionLost(self, reason)