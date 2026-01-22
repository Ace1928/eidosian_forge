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
class MulticastMixin:
    """
    Implement multicast functionality.
    """

    def getOutgoingInterface(self):
        i = self.socket.getsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF)
        return socket.inet_ntoa(struct.pack('@i', i))

    def setOutgoingInterface(self, addr):
        """
        Returns Deferred of success.
        """
        return self.reactor.resolve(addr).addCallback(self._setInterface)

    def _setInterface(self, addr):
        i = socket.inet_aton(addr)
        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, i)
        return 1

    def getLoopbackMode(self):
        return self.socket.getsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP)

    def setLoopbackMode(self, mode):
        mode = struct.pack('b', bool(mode))
        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, mode)

    def getTTL(self):
        return self.socket.getsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL)

    def setTTL(self, ttl):
        ttl = struct.pack('B', ttl)
        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)

    def joinGroup(self, addr, interface=''):
        """
        Join a multicast group. Returns Deferred of success.
        """
        return self.reactor.resolve(addr).addCallback(self._joinAddr1, interface, 1)

    def _joinAddr1(self, addr, interface, join):
        return self.reactor.resolve(interface).addCallback(self._joinAddr2, addr, join)

    def _joinAddr2(self, interface, addr, join):
        addr = socket.inet_aton(addr)
        interface = socket.inet_aton(interface)
        if join:
            cmd = socket.IP_ADD_MEMBERSHIP
        else:
            cmd = socket.IP_DROP_MEMBERSHIP
        try:
            self.socket.setsockopt(socket.IPPROTO_IP, cmd, addr + interface)
        except OSError as e:
            return failure.Failure(error.MulticastJoinError(addr, interface, *e.args))

    def leaveGroup(self, addr, interface=''):
        """
        Leave multicast group, return Deferred of success.
        """
        return self.reactor.resolve(addr).addCallback(self._joinAddr1, interface, 0)