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
@implementer(interfaces.IUNIXDatagramTransport)
class DatagramPort(_UNIXPort, udp.Port):
    """
    Datagram UNIX port, listening for packets.
    """
    addressFamily = socket.AF_UNIX

    def __init__(self, addr, proto, maxPacketSize=8192, mode=438, reactor=None):
        """Initialize with address to listen on."""
        udp.Port.__init__(self, addr, proto, maxPacketSize=maxPacketSize, reactor=reactor)
        self.mode = mode

    def __repr__(self) -> str:
        protocolName = reflect.qual(self.protocol.__class__)
        if hasattr(self, 'socket'):
            return f'<{protocolName} on {self.port!r}>'
        else:
            return f'<{protocolName} (not listening)>'

    def _bindSocket(self):
        log.msg(f'{self.protocol.__class__} starting on {repr(self.port)}')
        try:
            skt = self.createInternetSocket()
            if self.port:
                skt.bind(self.port)
        except OSError as le:
            raise error.CannotListenError(None, self.port, le)
        if self.port and _inFilesystemNamespace(self.port):
            os.chmod(self.port, self.mode)
        self.connected = 1
        self.socket = skt
        self.fileno = self.socket.fileno

    def write(self, datagram, address):
        """Write a datagram."""
        try:
            return self.socket.sendto(datagram, address)
        except OSError as se:
            no = se.args[0]
            if no == EINTR:
                return self.write(datagram, address)
            elif no == EMSGSIZE:
                raise error.MessageLengthError('message too long')
            elif no == EAGAIN:
                pass
            else:
                raise

    def connectionLost(self, reason=None):
        """Cleans up my socket."""
        log.msg('(Port %s Closed)' % repr(self.port))
        base.BasePort.connectionLost(self, reason)
        if hasattr(self, 'protocol'):
            self.protocol.doStop()
        self.connected = 0
        self.socket.close()
        del self.socket
        del self.fileno
        if hasattr(self, 'd'):
            self.d.callback(None)
            del self.d

    def setLogStr(self):
        self.logstr = reflect.qual(self.protocol.__class__) + ' (UDP)'