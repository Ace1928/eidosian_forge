import errno
import socket
import struct
import warnings
from typing import Optional
from zope.interface import implementer
from twisted.internet import address, defer, error, interfaces
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.internet.iocpreactor import abstract, iocpsupport as _iocp
from twisted.internet.iocpreactor.const import (
from twisted.internet.iocpreactor.interfaces import IReadWriteHandle
from twisted.python import failure, log
def _bindSocket(self):
    try:
        skt = self.createSocket()
        skt.bind((self.interface, self.port))
    except OSError as le:
        raise error.CannotListenError(self.interface, self.port, le)
    self._realPortNumber = skt.getsockname()[1]
    log.msg('%s starting on %s' % (self._getLogPrefix(self.protocol), self._realPortNumber))
    self.connected = True
    self.socket = skt
    self.getFileHandle = self.socket.fileno