from hashlib import md5
from os import close, fstat, stat, unlink, urandom
from pprint import pformat
from socket import AF_INET, SOCK_STREAM, SOL_SOCKET, socket
from stat import S_IMODE
from struct import pack
from tempfile import mkstemp, mktemp
from typing import Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from twisted.internet import base, interfaces
from twisted.internet.address import UNIXAddress
from twisted.internet.defer import Deferred, fail, gatherResults
from twisted.internet.endpoints import UNIXClientEndpoint, UNIXServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, DatagramProtocol, ServerFactory
from twisted.internet.task import LoopingCall
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import (
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.python.log import addObserver, err, removeObserver
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
class ListenUNIXMixin:
    """
    Mixin which uses L{IReactorTCP.listenUNIX} to hand out listening UNIX
    ports.
    """

    def getListeningPort(self, reactor, factory):
        """
        Get a UNIX port from a reactor
        """
        path = mktemp(suffix='.sock', dir='.')
        return reactor.listenUNIX(path, factory)

    def connectToListener(self, reactor, address, factory):
        """
        Connect to a listening UNIX socket.

        @param reactor: The reactor under test.
        @type reactor: L{IReactorUNIX}

        @param address: The listening's address.
        @type address: L{UNIXAddress}

        @param factory: The client factory.
        @type factory: L{ClientFactory}

        @return: The connector
        """
        return reactor.connectUNIX(address.name, factory)