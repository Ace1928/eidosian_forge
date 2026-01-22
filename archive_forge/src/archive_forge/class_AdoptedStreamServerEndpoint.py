import os
import re
import socket
import warnings
from typing import Optional, Sequence, Type
from unicodedata import normalize
from zope.interface import directlyProvides, implementer, provider
from constantly import NamedConstant, Names
from incremental import Version
from twisted.internet import defer, error, fdesc, interfaces, threads
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.internet.address import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, ProcessProtocol, Protocol
from twisted.internet._resolver import HostResolution
from twisted.internet.defer import Deferred
from twisted.internet.task import LoopingCall
from twisted.logger import Logger
from twisted.plugin import IPlugin, getPlugins
from twisted.python import deprecate, log
from twisted.python.compat import _matchingString, iterbytes, nativeString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.systemd import ListenFDs
from ._idna import _idnaBytes, _idnaText
@implementer(interfaces.IStreamServerEndpoint)
class AdoptedStreamServerEndpoint:
    """
    An endpoint for listening on a file descriptor initialized outside of
    Twisted.

    @ivar _used: A C{bool} indicating whether this endpoint has been used to
        listen with a factory yet.  C{True} if so.
    """
    _close = os.close
    _setNonBlocking = staticmethod(fdesc.setNonBlocking)

    def __init__(self, reactor, fileno, addressFamily):
        """
        @param reactor: An L{IReactorSocket} provider.

        @param fileno: An integer file descriptor corresponding to a listening
            I{SOCK_STREAM} socket.

        @param addressFamily: The address family of the socket given by
            C{fileno}.
        """
        self.reactor = reactor
        self.fileno = fileno
        self.addressFamily = addressFamily
        self._used = False

    def listen(self, factory):
        """
        Implement L{IStreamServerEndpoint.listen} to start listening on, and
        then close, C{self._fileno}.
        """
        if self._used:
            return defer.fail(error.AlreadyListened())
        self._used = True
        try:
            self._setNonBlocking(self.fileno)
            port = self.reactor.adoptStreamPort(self.fileno, self.addressFamily, factory)
            self._close(self.fileno)
        except BaseException:
            return defer.fail()
        return defer.succeed(port)