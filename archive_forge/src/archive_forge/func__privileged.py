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
def _privileged(original):
    """
    Wrap a L{MemoryIOSystem} method with permission-checking logic.  The
    returned function will check C{self.permissions} and raise L{IOError} with
    L{errno.EPERM} if the function name is not listed as an available
    permission.

    @param original: The L{MemoryIOSystem} instance to wrap.

    @return: A wrapper around C{original} that applies permission checks.
    """

    @wraps(original)
    def permissionChecker(self, *args, **kwargs):
        if original.__name__ not in self.permissions:
            raise OSError(EPERM, 'Operation not permitted')
        return original(self, *args, **kwargs)
    return permissionChecker