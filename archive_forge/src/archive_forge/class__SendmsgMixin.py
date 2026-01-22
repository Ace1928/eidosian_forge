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
class _SendmsgMixin:
    """
    Mixin for stream-oriented UNIX transports which uses sendmsg and recvmsg to
    offer additional functionality, such as copying file descriptors into other
    processes.

    @ivar _writeSomeDataBase: The class which provides the basic implementation
        of C{writeSomeData}.  Ultimately this should be a subclass of
        L{twisted.internet.abstract.FileDescriptor}.  Subclasses which mix in
        L{_SendmsgMixin} must define this.

    @ivar _sendmsgQueue: A C{list} of C{int} holding file descriptors which are
        currently buffered before being sent.

    @ivar _fileDescriptorBufferSize: An C{int} giving the maximum number of file
        descriptors to accept and queue for sending before pausing the
        registered producer, if there is one.
    """
    _writeSomeDataBase: Optional[Type[FileDescriptor]] = None
    _fileDescriptorBufferSize = 64

    def __init__(self):
        self._sendmsgQueue = []

    def _isSendBufferFull(self):
        """
        Determine whether the user-space send buffer for this transport is full
        or not.

        This extends the base determination by adding consideration of how many
        file descriptors need to be sent using L{sendmsg.sendmsg}.  When there
        are more than C{self._fileDescriptorBufferSize}, the buffer is
        considered full.

        @return: C{True} if it is full, C{False} otherwise.
        """
        return len(self._sendmsgQueue) > self._fileDescriptorBufferSize or self._writeSomeDataBase._isSendBufferFull(self)

    def sendFileDescriptor(self, fileno):
        """
        Queue the given file descriptor to be sent and start trying to send it.
        """
        self._sendmsgQueue.append(fileno)
        self._maybePauseProducer()
        self.startWriting()

    def writeSomeData(self, data):
        """
        Send as much of C{data} as possible.  Also send any pending file
        descriptors.
        """
        if len(self._sendmsgQueue) > len(data):
            return error.FileDescriptorOverrun()
        index = 0
        try:
            while index < len(self._sendmsgQueue):
                fd = self._sendmsgQueue[index]
                try:
                    untilConcludes(sendmsg.sendmsg, self.socket, data[index:index + 1], _ancillaryDescriptor(fd))
                except OSError as se:
                    if se.args[0] in (EWOULDBLOCK, ENOBUFS):
                        return index
                    else:
                        return main.CONNECTION_LOST
                else:
                    index += 1
        finally:
            del self._sendmsgQueue[:index]
        limitedData = lazyByteSlice(data, index)
        result = self._writeSomeDataBase.writeSomeData(self, limitedData)
        try:
            return index + result
        except TypeError:
            return result

    def doRead(self):
        """
        Calls {IProtocol.dataReceived} with all available data and
        L{IFileDescriptorReceiver.fileDescriptorReceived} once for each
        received file descriptor in ancillary data.

        This reads up to C{self.bufferSize} bytes of data from its socket, then
        dispatches the data to protocol callbacks to be handled.  If the
        connection is not lost through an error in the underlying recvmsg(),
        this function will return the result of the dataReceived call.
        """
        try:
            data, ancillary, flags = untilConcludes(sendmsg.recvmsg, self.socket, self.bufferSize)
        except OSError as se:
            if se.args[0] == EWOULDBLOCK:
                return
            else:
                return main.CONNECTION_LOST
        for cmsgLevel, cmsgType, cmsgData in ancillary:
            if cmsgLevel == socket.SOL_SOCKET and cmsgType == sendmsg.SCM_RIGHTS:
                self._ancillaryLevelSOLSOCKETTypeSCMRIGHTS(cmsgData)
            else:
                log.msg(format='%(protocolName)s (on %(hostAddress)r) received unsupported ancillary data (level=%(cmsgLevel)r, type=%(cmsgType)r) from %(peerAddress)r.', hostAddress=self.getHost(), peerAddress=self.getPeer(), protocolName=self._getLogPrefix(self.protocol), cmsgLevel=cmsgLevel, cmsgType=cmsgType)
        return self._dataReceived(data)

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