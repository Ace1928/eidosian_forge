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
def _getrealname(addr):
    """
    Return a 2-tuple of socket IP and port for IPv4 and a 4-tuple of
    socket IP, port, flowInfo, and scopeID for IPv6.  For IPv6, it
    returns the interface portion (the part after the %) as a part of
    the IPv6 address, which Python 3.7+ does not include.

    @param addr: A 2-tuple for IPv4 information or a 4-tuple for IPv6
        information.
    """
    if len(addr) == 4:
        host = socket.getnameinfo(addr, socket.NI_NUMERICHOST | socket.NI_NUMERICSERV)[0]
        return tuple([host] + list(addr[1:]))
    else:
        return addr[:2]