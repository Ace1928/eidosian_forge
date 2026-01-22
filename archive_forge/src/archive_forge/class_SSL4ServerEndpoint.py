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
class SSL4ServerEndpoint:
    """
    SSL secured TCP server endpoint with an IPv4 configuration.
    """

    def __init__(self, reactor, port, sslContextFactory, backlog=50, interface=''):
        """
        @param reactor: An L{IReactorSSL} provider.

        @param port: The port number used for listening
        @type port: int

        @param sslContextFactory: An instance of
            L{interfaces.IOpenSSLContextFactory}.

        @param backlog: Size of the listen queue
        @type backlog: int

        @param interface: The hostname to bind to, defaults to '' (all)
        @type interface: str
        """
        self._reactor = reactor
        self._port = port
        self._sslContextFactory = sslContextFactory
        self._backlog = backlog
        self._interface = interface

    def listen(self, protocolFactory):
        """
        Implement L{IStreamServerEndpoint.listen} to listen for SSL on a
        TCP socket.
        """
        return defer.execute(self._reactor.listenSSL, self._port, protocolFactory, contextFactory=self._sslContextFactory, backlog=self._backlog, interface=self._interface)