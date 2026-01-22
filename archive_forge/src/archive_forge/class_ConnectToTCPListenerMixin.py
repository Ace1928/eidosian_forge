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
class ConnectToTCPListenerMixin:
    """
    Provides L{connectToListener} for TCP transports.

    @ivar LISTENER_HOST: The host on which the port is expected to be
        listening.  This is specific to avoid compatibility issues
        with Windows, which cannot connect to the wildcard host.
    @type LISTENER_HOST: L{str}

    @see: U{http://twistedmatrix.com/trac/ticket/1472}
    """
    LISTENER_HOST = '127.0.0.1'

    def connectToListener(self, reactor, address, factory):
        """
        Connect to the given listening TCP port.

        @param reactor: The reactor under test.
        @type reactor: L{IReactorTCP}

        @param address: The listening port's address.  Only the
            C{port} component is used; see L{LISTENER_HOST}.
        @type address: L{IPv4Address} or L{IPv6Address}

        @param factory: The client factory.
        @type factory: L{ClientFactory}

        @return: The connector
        """
        return reactor.connectTCP(self.LISTENER_HOST, address.port, factory)