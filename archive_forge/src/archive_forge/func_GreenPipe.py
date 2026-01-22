import _pyio as _original_pyio
import errno
import os as _original_os
import socket as _original_socket
from io import (
from types import FunctionType
from eventlet.greenio.base import (
from eventlet.hubs import notify_close, notify_opened, IOClosed, trampoline
from eventlet.support import get_errno
def GreenPipe(name, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    try:
        fileno = name.fileno()
    except AttributeError:
        pass
    else:
        fileno = _original_os.dup(fileno)
        name.close()
        name = fileno
    return _open(name, mode, buffering, encoding, errors, newline, closefd, opener)