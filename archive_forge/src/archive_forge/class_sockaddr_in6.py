import socket
import sys
from ctypes import (
from ctypes.util import find_library
from socket import AF_INET, AF_INET6, inet_ntop
from typing import Any, List, Tuple
from twisted.python.compat import nativeString
class sockaddr_in6(Structure):
    _fields_ = _sockaddrCommon + [('sin_port', c_ushort), ('sin_flowinfo', c_uint32), ('sin_addr', in6_addr)]