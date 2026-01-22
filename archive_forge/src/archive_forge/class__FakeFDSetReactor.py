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
@implementer(IReactorFDSet)
class _FakeFDSetReactor:
    """
    An in-memory implementation of L{IReactorFDSet}, which records the current
    sets of active L{IReadDescriptor} and L{IWriteDescriptor}s.

    @ivar _readers: The set of L{IReadDescriptor}s active on this
        L{_FakeFDSetReactor}
    @type _readers: L{set}

    @ivar _writers: The set of L{IWriteDescriptor}s active on this
        L{_FakeFDSetReactor}
    @ivar _writers: L{set}
    """

    def __init__(self):
        self._readers = set()
        self._writers = set()

    def addReader(self, reader):
        self._readers.add(reader)

    def removeReader(self, reader):
        if reader in self._readers:
            self._readers.remove(reader)

    def addWriter(self, writer):
        self._writers.add(writer)

    def removeWriter(self, writer):
        if writer in self._writers:
            self._writers.remove(writer)

    def removeAll(self):
        result = self.getReaders() + self.getWriters()
        self.__init__()
        return result

    def getReaders(self):
        return list(self._readers)

    def getWriters(self):
        return list(self._writers)