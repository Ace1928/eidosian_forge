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
def _setRealAddress(self, address):
    """
        Set the resolved address of this L{_BaseBaseClient} and initiate the
        connection attempt.

        @param address: Depending on whether this is an IPv4 or IPv6 connection
            attempt, a 2-tuple of C{(host, port)} or a 4-tuple of C{(host,
            port, flow, scope)}.  At this point it is a fully resolved address,
            and the 'host' portion will always be an IP address, not a DNS
            name.
        """
    if len(address) == 4:
        hostname = socket.getnameinfo(address, socket.NI_NUMERICHOST | socket.NI_NUMERICSERV)[0]
        self.realAddress = tuple([hostname] + list(address[1:]))
    else:
        self.realAddress = address
    self.doConnect()