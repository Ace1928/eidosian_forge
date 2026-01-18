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
@skipIf(ipv6Skip, ipv6SkipReason)
def test_portGetHostOnIPv6(self):
    """
        When listening on an IPv6 address, L{IListeningPort.getHost} returns
        an L{IPv6Address} with C{host} and C{port} attributes reflecting the
        address the port is bound to.
        """
    reactor = self.buildReactor()
    host, portNumber = findFreePort(family=socket.AF_INET6, interface='::1')[:2]
    port = self.getListeningPort(reactor, ServerFactory(), portNumber, host)
    address = port.getHost()
    self.assertIsInstance(address, IPv6Address)
    self.assertEqual('::1', address.host)
    self.assertEqual(portNumber, address.port)