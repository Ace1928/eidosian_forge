from __future__ import annotations
import os
import socket
import struct
import sys
from typing import Callable, ClassVar, List, Optional, Union
from zope.interface import Interface, implementer
import attr
import typing_extensions
from twisted.internet.interfaces import (
from twisted.logger import ILogObserver, LogEvent, Logger
from twisted.python import deprecate, versions
from twisted.python.compat import lazyByteSlice
from twisted.python.runtime import platformType
from errno import errorcode
from twisted.internet import abstract, address, base, error, fdesc, main
from twisted.internet.error import CannotListenError
from twisted.internet.protocol import Protocol
from twisted.internet.task import deferLater
from twisted.python import failure, log, reflect
from twisted.python.util import untilConcludes
class _BaseTCPClient:
    """
    Code shared with other (non-POSIX) reactors for management of outgoing TCP
    connections (both TCPv4 and TCPv6).

    @note: In order to be functional, this class must be mixed into the same
        hierarchy as L{_BaseBaseClient}.  It would subclass L{_BaseBaseClient}
        directly, but the class hierarchy here is divided in strange ways out
        of the need to share code along multiple axes; specifically, with the
        IOCP reactor and also with UNIX clients in other reactors.

    @ivar _addressType: The Twisted _IPAddress implementation for this client
    @type _addressType: L{IPv4Address} or L{IPv6Address}

    @ivar connector: The L{Connector} which is driving this L{_BaseTCPClient}'s
        connection attempt.

    @ivar addr: The address that this socket will be connecting to.
    @type addr: If IPv4, a 2-C{tuple} of C{(str host, int port)}.  If IPv6, a
        4-C{tuple} of (C{str host, int port, int ignored, int scope}).

    @ivar createInternetSocket: Subclasses must implement this as a method to
        create a python socket object of the appropriate address family and
        socket type.
    @type createInternetSocket: 0-argument callable returning
        C{socket._socketobject}.
    """
    _addressType = address.IPv4Address

    def __init__(self, host, port, bindAddress, connector, reactor=None):
        self.connector = connector
        self.addr = (host, port)
        whenDone = self.resolveAddress
        err = None
        skt = None
        if abstract.isIPAddress(host):
            self._requiresResolution = False
        elif abstract.isIPv6Address(host):
            self._requiresResolution = False
            self.addr = _resolveIPv6(host, port)
            self.addressFamily = socket.AF_INET6
            self._addressType = address.IPv6Address
        else:
            self._requiresResolution = True
        try:
            skt = self.createInternetSocket()
        except OSError as se:
            err = error.ConnectBindError(se.args[0], se.args[1])
            whenDone = None
        if whenDone and bindAddress is not None:
            try:
                if abstract.isIPv6Address(bindAddress[0]):
                    bindinfo = _resolveIPv6(*bindAddress)
                else:
                    bindinfo = bindAddress
                skt.bind(bindinfo)
            except OSError as se:
                err = error.ConnectBindError(se.args[0], se.args[1])
                whenDone = None
        self._finishInit(whenDone, skt, err, reactor)

    def getHost(self):
        """
        Returns an L{IPv4Address} or L{IPv6Address}.

        This indicates the address from which I am connecting.
        """
        return self._addressType('TCP', *_getsockname(self.socket))

    def getPeer(self):
        """
        Returns an L{IPv4Address} or L{IPv6Address}.

        This indicates the address that I am connected to.
        """
        return self._addressType('TCP', *self.realAddress)

    def __repr__(self) -> str:
        s = f'<{self.__class__} to {self.addr} at {id(self):x}>'
        return s