import socket
import struct
from collections import deque
from errno import EAGAIN, EBADF, EINTR, EINVAL, ENOBUFS, ENOSYS, EPERM, EWOULDBLOCK
from functools import wraps
from zope.interface import implementer
from twisted.internet.protocol import DatagramProtocol
from twisted.pair.ethernet import EthernetProtocol
from twisted.pair.ip import IPProtocol
from twisted.pair.rawudp import RawUDPProtocol
from twisted.pair.tuntap import _IFNAMSIZ, _TUNSETIFF, TunnelFlags, _IInputOutputSystem
from twisted.python.compat import nativeString
class _FakePort:
    """
    A socket-like object which can be used to read UDP datagrams from
    tunnel-like file descriptors managed by a L{MemoryIOSystem}.
    """

    def __init__(self, system, fileno):
        self._system = system
        self._fileno = fileno

    def recv(self, nbytes):
        """
        Receive a datagram sent to this port using the L{MemoryIOSystem} which
        created this object.

        This behaves like L{socket.socket.recv} but the data being I{sent} and
        I{received} only passes through various memory buffers managed by this
        object and L{MemoryIOSystem}.

        @see: L{socket.socket.recv}
        """
        data = self._system._openFiles[self._fileno].writeBuffer.popleft()
        datagrams = []
        receiver = DatagramProtocol()

        def capture(datagram, address):
            datagrams.append(datagram)
        receiver.datagramReceived = capture
        udp = RawUDPProtocol()
        udp.addProto(12345, receiver)
        ip = IPProtocol()
        ip.addProto(17, udp)
        mode = self._system._openFiles[self._fileno].tunnelMode
        if mode & TunnelFlags.IFF_TAP.value:
            ether = EthernetProtocol()
            ether.addProto(2048, ip)
            datagramReceived = ether.datagramReceived
        else:
            datagramReceived = lambda data: ip.datagramReceived(data, None, None, None, None)
        dataHasPI = not mode & TunnelFlags.IFF_NO_PI.value
        if dataHasPI:
            data = data[_PI_SIZE:]
        datagramReceived(data)
        return datagrams[0][:nbytes]