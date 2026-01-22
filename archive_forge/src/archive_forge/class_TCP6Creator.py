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
class TCP6Creator(TCPCreator):
    """
    Create IPv6 TCP endpoints for
    C{ReactorBuilder.runProtocolsWithReactor}-based tests.

    The endpoint types in question here are still the TCP4 variety, since
    these simply pass through IPv6 address literals to the reactor, and we are
    only testing address literals, not name resolution (as name resolution has
    not yet been implemented).  See http://twistedmatrix.com/trac/ticket/4470
    for more specific information about new endpoint classes.  The naming is
    slightly misleading, but presumably if you're passing an IPv6 literal, you
    know what you're asking for.
    """

    def __init__(self):
        self.interface = getLinkLocalIPv6Address()