import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def _asFileDescriptor(obj):
    fd = None
    if not isinstance(obj, int):
        meth = getattr(obj, 'fileno', None)
        if meth is not None:
            obj = meth()
    if isinstance(obj, int):
        fd = obj
    if not isinstance(fd, int):
        raise TypeError('argument must be an int, or have a fileno() method.')
    elif fd < 0:
        raise ValueError('file descriptor cannot be a negative integer (%i)' % (fd,))
    return fd