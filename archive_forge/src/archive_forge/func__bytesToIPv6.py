import binascii
import struct
from typing import Callable, Tuple, Type, Union
from zope.interface import implementer
from constantly import ValueConstant, Values
from typing_extensions import Literal
from twisted.internet import address
from twisted.python import compat
from . import _info, _interfaces
from ._exceptions import (
@staticmethod
def _bytesToIPv6(bytestring: bytes) -> bytes:
    """
        Convert packed 128-bit IPv6 address bytes into a colon-separated ASCII
        bytes representation of that address.

        @param bytestring: 16 octets representing an IPv6 address.
        @type bytestring: L{bytes}

        @return: a dotted-quad notation IPv6 address.
        @rtype: L{bytes}
        """
    hexString = binascii.b2a_hex(bytestring)
    return b':'.join((f'{int(hexString[b:b + 4], 16):x}'.encode('ascii') for b in range(0, 32, 4)))