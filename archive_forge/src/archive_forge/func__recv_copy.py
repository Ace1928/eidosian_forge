from __future__ import annotations
import time
import warnings
from threading import Event
from weakref import ref
import cython as C
from cython import (
from cython.cimports.cpython import (
from cython.cimports.libc.errno import EAGAIN, EINTR, ENAMETOOLONG, ENOENT, ENOTSOCK
from cython.cimports.libc.stdint import uint32_t
from cython.cimports.libc.stdio import fprintf
from cython.cimports.libc.stdio import stderr as cstderr
from cython.cimports.libc.stdlib import free, malloc
from cython.cimports.libc.string import memcpy
from cython.cimports.zmq.backend.cython._externs import (
from cython.cimports.zmq.backend.cython.libzmq import (
from cython.cimports.zmq.backend.cython.libzmq import zmq_errno as _zmq_errno
from cython.cimports.zmq.backend.cython.libzmq import zmq_poll as zmq_poll_c
from cython.cimports.zmq.utils.buffers import asbuffer_r
import zmq
from zmq.constants import SocketOption, _OptType
from zmq.error import InterruptedSystemCall, ZMQError, _check_version
@cfunc
@inline
def _recv_copy(handle: p_void, flags: C.int=0):
    """Receive a message and return a copy"""
    zmq_msg = declare(zmq_msg_t)
    zmq_msg_p: pointer(zmq_msg_t) = address(zmq_msg)
    rc: C.int = zmq_msg_init(zmq_msg_p)
    _check_rc(rc)
    while True:
        with nogil:
            rc = zmq_msg_recv(zmq_msg_p, handle, flags)
        try:
            _check_rc(rc)
        except InterruptedSystemCall:
            continue
        except Exception:
            zmq_msg_close(zmq_msg_p)
            raise
        else:
            break
    msg_bytes = _copy_zmq_msg_bytes(zmq_msg_p)
    zmq_msg_close(zmq_msg_p)
    return msg_bytes