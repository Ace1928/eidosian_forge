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
def _resolveIPv6(ip, port):
    """
    Resolve an IPv6 literal into an IPv6 address.

    This is necessary to resolve any embedded scope identifiers to the relevant
    C{sin6_scope_id} for use with C{socket.connect()}, C{socket.listen()}, or
    C{socket.bind()}; see U{RFC 3493 <https://tools.ietf.org/html/rfc3493>} for
    more information.

    @param ip: An IPv6 address literal.
    @type ip: C{str}

    @param port: A port number.
    @type port: C{int}

    @return: a 4-tuple of C{(host, port, flow, scope)}, suitable for use as an
        IPv6 address.

    @raise socket.gaierror: if either the IP or port is not numeric as it
        should be.
    """
    return socket.getaddrinfo(ip, port, 0, 0, 0, _NUMERIC_ONLY)[0][4]