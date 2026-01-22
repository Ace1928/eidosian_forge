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
def _setAddressFamily(self):
    """
        Resolve address family for the socket.
        """
    if isIPv6Address(self.interface):
        self.addressFamily = socket.AF_INET6
    elif isIPAddress(self.interface):
        self.addressFamily = socket.AF_INET
    elif self.interface:
        raise error.InvalidAddressError(self.interface, 'not an IPv4 or IPv6 address')