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
def addToReadBuffer(self, datagram):
    """
        Deliver a datagram to this tunnel's read buffer.  This makes it
        available to be read later using the C{read} method.

        @param datagram: The IPv4 datagram to deliver.  If the mode of this
            tunnel is TAP then ethernet framing will be added automatically.
        @type datagram: L{bytes}
        """
    if self.tunnelMode & TunnelFlags.IFF_TAP.value:
        datagram = _ethernet(src=b'\x00' * 6, dst=b'\xff' * 6, protocol=_IPv4, payload=datagram)
    self.readBuffer.append(datagram)