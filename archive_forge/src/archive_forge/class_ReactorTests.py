from __future__ import annotations
from typing import Callable
from zope.interface.verify import verifyObject
from typing_extensions import Protocol
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory
from twisted.internet.testing import (
from twisted.python.reflect import namedAny
from twisted.trial.unittest import TestCase
class ReactorTests(TestCase):
    """
    Tests for L{MemoryReactor} and L{RaisingMemoryReactor}.
    """

    def test_memoryReactorProvides(self) -> None:
        """
        L{MemoryReactor} provides all of the attributes described by the
        interfaces it advertises.
        """
        memoryReactor = MemoryReactor()
        verifyObject(IReactorTCP, memoryReactor)
        verifyObject(IReactorSSL, memoryReactor)
        verifyObject(IReactorUNIX, memoryReactor)

    def test_raisingReactorProvides(self) -> None:
        """
        L{RaisingMemoryReactor} provides all of the attributes described by the
        interfaces it advertises.
        """
        raisingReactor = RaisingMemoryReactor()
        verifyObject(IReactorTCP, raisingReactor)
        verifyObject(IReactorSSL, raisingReactor)
        verifyObject(IReactorUNIX, raisingReactor)

    def test_connectDestination(self) -> None:
        """
        L{MemoryReactor.connectTCP}, L{MemoryReactor.connectSSL}, and
        L{MemoryReactor.connectUNIX} will return an L{IConnector} whose
        C{getDestination} method returns an L{IAddress} with attributes which
        reflect the values passed.
        """
        memoryReactor = MemoryReactor()
        for connector in [memoryReactor.connectTCP('test.example.com', 8321, ClientFactory()), memoryReactor.connectSSL('test.example.com', 8321, ClientFactory(), None)]:
            verifyObject(IConnector, connector)
            address = connector.getDestination()
            verifyObject(IAddress, address)
            self.assertEqual(address.host, 'test.example.com')
            self.assertEqual(address.port, 8321)
        connector = memoryReactor.connectUNIX(b'/fake/path', ClientFactory())
        verifyObject(IConnector, connector)
        address = connector.getDestination()
        verifyObject(IAddress, address)
        self.assertEqual(address.name, b'/fake/path')

    def test_listenDefaultHost(self) -> None:
        """
        L{MemoryReactor.listenTCP}, L{MemoryReactor.listenSSL} and
        L{MemoryReactor.listenUNIX} will return an L{IListeningPort} whose
        C{getHost} method returns an L{IAddress}; C{listenTCP} and C{listenSSL}
        will have a default host of C{'0.0.0.0'}, and a port that reflects the
        value passed, and C{listenUNIX} will have a name that reflects the path
        passed.
        """
        memoryReactor = MemoryReactor()
        for port in [memoryReactor.listenTCP(8242, Factory()), memoryReactor.listenSSL(8242, Factory(), None)]:
            verifyObject(IListeningPort, port)
            address = port.getHost()
            verifyObject(IAddress, address)
            self.assertEqual(address.host, '0.0.0.0')
            self.assertEqual(address.port, 8242)
        port = memoryReactor.listenUNIX(b'/path/to/socket', Factory())
        verifyObject(IListeningPort, port)
        address = port.getHost()
        verifyObject(IAddress, address)
        self.assertEqual(address.name, b'/path/to/socket')

    def test_readers(self) -> None:
        """
        Adding, removing, and listing readers works.
        """
        reader = object()
        reactor = MemoryReactor()
        reactor.addReader(reader)
        reactor.addReader(reader)
        self.assertEqual(reactor.getReaders(), [reader])
        reactor.removeReader(reader)
        self.assertEqual(reactor.getReaders(), [])

    def test_writers(self) -> None:
        """
        Adding, removing, and listing writers works.
        """
        writer = object()
        reactor = MemoryReactor()
        reactor.addWriter(writer)
        reactor.addWriter(writer)
        self.assertEqual(reactor.getWriters(), [writer])
        reactor.removeWriter(writer)
        self.assertEqual(reactor.getWriters(), [])