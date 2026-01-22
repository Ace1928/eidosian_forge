import os
import socket
import stat
import struct
from errno import EAGAIN, ECONNREFUSED, EINTR, EMSGSIZE, ENOBUFS, EWOULDBLOCK
from typing import Optional, Type
from zope.interface import implementedBy, implementer, implementer_only
from twisted.internet import address, base, error, interfaces, main, protocol, tcp, udp
from twisted.internet.abstract import FileDescriptor
from twisted.python import failure, lockfile, log, reflect
from twisted.python.compat import lazyByteSlice
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.python.util import untilConcludes
def _ancillaryDescriptor(fd):
    """
    Pack an integer into an ancillary data structure suitable for use with
    L{sendmsg.sendmsg}.
    """
    packed = struct.pack('i', fd)
    return [(socket.SOL_SOCKET, sendmsg.SCM_RIGHTS, packed)]