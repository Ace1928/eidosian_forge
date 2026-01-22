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
class _SocketCloser:
    """
    @ivar _shouldShutdown: Set to C{True} if C{shutdown} should be called
        before calling C{close} on the underlying socket.
    @type _shouldShutdown: C{bool}
    """
    _shouldShutdown = True

    def _closeSocket(self, orderly):
        skt = self.socket
        try:
            if orderly:
                if self._shouldShutdown:
                    skt.shutdown(2)
            else:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
        except OSError:
            pass
        try:
            skt.close()
        except OSError:
            pass