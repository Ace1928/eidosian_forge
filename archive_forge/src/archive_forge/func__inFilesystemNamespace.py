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
def _inFilesystemNamespace(path):
    """
    Determine whether the given unix socket path is in a filesystem namespace.

    While most PF_UNIX sockets are entries in the filesystem, Linux 2.2 and
    above support PF_UNIX sockets in an "abstract namespace" that does not
    correspond to any path. This function returns C{True} if the given socket
    path is stored in the filesystem and C{False} if the path is in this
    abstract namespace.
    """
    return path[:1] not in (b'\x00', '\x00')