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
@implementer(interfaces.IStreamClientEndpoint)
class SSL4ClientEndpoint:
    """
    SSL secured TCP client endpoint with an IPv4 configuration
    """

    def __init__(self, reactor, host, port, sslContextFactory, timeout=30, bindAddress=None):
        """
        @param reactor: An L{IReactorSSL} provider.

        @param host: A hostname, used when connecting
        @type host: str

        @param port: The port number, used when connecting
        @type port: int

        @param sslContextFactory: SSL Configuration information as an instance
            of L{interfaces.IOpenSSLContextFactory}.

        @param timeout: Number of seconds to wait before assuming the
            connection has failed.
        @type timeout: int

        @param bindAddress: A (host, port) tuple of local address to bind to,
            or None.
        @type bindAddress: tuple
        """
        self._reactor = reactor
        self._host = host
        self._port = port
        self._sslContextFactory = sslContextFactory
        self._timeout = timeout
        self._bindAddress = bindAddress

    def connect(self, protocolFactory):
        """
        Implement L{IStreamClientEndpoint.connect} to connect with SSL over
        TCP.
        """
        try:
            wf = _WrappingFactory(protocolFactory)
            self._reactor.connectSSL(self._host, self._port, wf, self._sslContextFactory, timeout=self._timeout, bindAddress=self._bindAddress)
            return wf._onConnection
        except BaseException:
            return defer.fail()