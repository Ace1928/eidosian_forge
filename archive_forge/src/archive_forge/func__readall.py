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
def _readall(self, file, count):
    """Receive EXACTLY the number of bytes requested from the file object.

        Blocks until the required number of bytes have been received."""
    data = b''
    while len(data) < count:
        d = file.read(count - len(data))
        if not d:
            raise GeneralProxyError('Connection closed unexpectedly')
        data += d
    return data