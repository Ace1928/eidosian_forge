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
def _ancillaryLevelSOLSOCKETTypeSCMRIGHTS(self, cmsgData):
    """
        Processes ancillary data with level SOL_SOCKET and type SCM_RIGHTS,
        indicating that the ancillary data payload holds file descriptors.

        Calls L{IFileDescriptorReceiver.fileDescriptorReceived} once for each
        received file descriptor or logs a message if the protocol does not
        implement L{IFileDescriptorReceiver}.

        @param cmsgData: Ancillary data payload.
        @type cmsgData: L{bytes}
        """
    fdCount = len(cmsgData) // 4
    fds = struct.unpack('i' * fdCount, cmsgData)
    if interfaces.IFileDescriptorReceiver.providedBy(self.protocol):
        for fd in fds:
            self.protocol.fileDescriptorReceived(fd)
    else:
        log.msg(format='%(protocolName)s (on %(hostAddress)r) does not provide IFileDescriptorReceiver; closing file descriptor received (from %(peerAddress)r).', hostAddress=self.getHost(), peerAddress=self.getPeer(), protocolName=self._getLogPrefix(self.protocol))
        for fd in fds:
            os.close(fd)