from __future__ import annotations
from io import BytesIO
from socket import AF_INET, AF_INET6
from typing import Callable, Iterator, Sequence, overload
from zope.interface import implementedBy, implementer
from zope.interface.verify import verifyClass
from typing_extensions import ParamSpec, Self
from twisted.internet import address, error, protocol, task
from twisted.internet.abstract import _dataMustBeBytes, isIPv6Address
from twisted.internet.address import IPv4Address, IPv6Address, UNIXAddress
from twisted.internet.defer import Deferred
from twisted.internet.error import UnsupportedAddressFamily
from twisted.internet.interfaces import (
from twisted.internet.task import Clock
from twisted.logger import ILogObserver, LogEvent, LogPublisher
from twisted.protocols import basic
from twisted.python import failure
from twisted.trial.unittest import TestCase
@implementer(IReactorTCP, IReactorSSL, IReactorUNIX, IReactorSocket)
class RaisingMemoryReactor:
    """
    A fake reactor to be used in tests.  It accepts TCP connection setup
    attempts, but they will fail.

    @ivar _listenException: An instance of an L{Exception}
    @ivar _connectException: An instance of an L{Exception}
    """

    def __init__(self, listenException=None, connectException=None):
        """
        @param listenException: An instance of an L{Exception} to raise
            when any C{listen} method is called.

        @param connectException: An instance of an L{Exception} to raise
            when any C{connect} method is called.
        """
        self._listenException = listenException
        self._connectException = connectException

    def adoptStreamPort(self, fileno, addressFamily, factory):
        """
        Fake L{IReactorSocket.adoptStreamPort}, that raises
        L{_listenException}.
        """
        raise self._listenException

    def listenTCP(self, port, factory, backlog=50, interface=''):
        """
        Fake L{IReactorTCP.listenTCP}, that raises L{_listenException}.
        """
        raise self._listenException

    def connectTCP(self, host, port, factory, timeout=30, bindAddress=None):
        """
        Fake L{IReactorTCP.connectTCP}, that raises L{_connectException}.
        """
        raise self._connectException

    def listenSSL(self, port, factory, contextFactory, backlog=50, interface=''):
        """
        Fake L{IReactorSSL.listenSSL}, that raises L{_listenException}.
        """
        raise self._listenException

    def connectSSL(self, host, port, factory, contextFactory, timeout=30, bindAddress=None):
        """
        Fake L{IReactorSSL.connectSSL}, that raises L{_connectException}.
        """
        raise self._connectException

    def listenUNIX(self, address, factory, backlog=50, mode=438, wantPID=0):
        """
        Fake L{IReactorUNIX.listenUNIX}, that raises L{_listenException}.
        """
        raise self._listenException

    def connectUNIX(self, address, factory, timeout=30, checkPID=0):
        """
        Fake L{IReactorUNIX.connectUNIX}, that raises L{_connectException}.
        """
        raise self._connectException

    def adoptDatagramPort(self, fileDescriptor, addressFamily, protocol, maxPacketSize):
        """
        Fake L{IReactorSocket.adoptDatagramPort}, that raises
        L{_connectException}.
        """
        raise self._connectException

    def adoptStreamConnection(self, fileDescriptor, addressFamily, factory):
        """
        Fake L{IReactorSocket.adoptStreamConnection}, that raises
        L{_connectException}.
        """
        raise self._connectException