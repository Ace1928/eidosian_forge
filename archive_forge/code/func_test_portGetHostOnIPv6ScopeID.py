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
def test_portGetHostOnIPv6ScopeID(self):
    """
        When a link-local IPv6 address including a scope identifier is passed
        as the C{interface} argument to L{IReactorTCP.listenTCP}, the resulting
        L{IListeningPort} reports its address as an L{IPv6Address} with a host
        value that includes the scope identifier.
        """
    linkLocal = getLinkLocalIPv6Address()
    reactor = self.buildReactor()
    port = self.getListeningPort(reactor, ServerFactory(), 0, linkLocal)
    address = port.getHost()
    self.assertIsInstance(address, IPv6Address)
    self.assertEqual(linkLocal, address.host)