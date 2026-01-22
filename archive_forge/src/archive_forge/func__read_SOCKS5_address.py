from base64 import b64encode
import six
from errno import EOPNOTSUPP, EINVAL, EAGAIN
import functools
from io import BytesIO
import logging
import os
from os import SEEK_CUR
import socket
import struct
import sys
def _read_SOCKS5_address(self, file):
    atyp = self._readall(file, 1)
    if atyp == b'\x01':
        addr = socket.inet_ntoa(self._readall(file, 4))
    elif atyp == b'\x03':
        length = self._readall(file, 1)
        addr = self._readall(file, ord(length))
    elif atyp == b'\x04':
        addr = socket.inet_ntop(socket.AF_INET6, self._readall(file, 16))
    else:
        raise GeneralProxyError('SOCKS5 proxy server sent invalid data')
    port = struct.unpack('>H', self._readall(file, 2))[0]
    return (addr, port)