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
@nogil
def _mq_inline(in_socket: p_void, out_socket: p_void, side_socket: p_void, in_msg_ptr: pointer(zmq_msg_t), out_msg_ptr: pointer(zmq_msg_t), swap_ids: bint) -> C.int:
    """
    inner C function for monitored_queue
    """
    msg: zmq_msg_t = declare(zmq_msg_t)
    rc: C.int = zmq_msg_init(address(msg))
    id_msg = declare(zmq_msg_t)
    rc = zmq_msg_init(address(id_msg))
    if rc < 0:
        return rc
    side_msg = declare(zmq_msg_t)
    rc = zmq_msg_init(address(side_msg))
    if rc < 0:
        return rc
    items = declare(zmq_pollitem_t[2])
    items[0].socket = in_socket
    items[0].events = ZMQ_POLLIN
    items[0].fd = items[0].revents = 0
    items[1].socket = out_socket
    items[1].events = ZMQ_POLLIN
    items[1].fd = items[1].revents = 0
    while True:
        rc = zmq_poll_c(address(items[0]), 2, -1)
        if rc < 0:
            return rc
        if items[0].revents & ZMQ_POLLIN:
            rc = zmq_msg_copy(address(side_msg), in_msg_ptr)
            if rc < 0:
                return rc
            rc = zmq_msg_send(address(side_msg), side_socket, ZMQ_SNDMORE)
            if rc < 0:
                return rc
            rc = _mq_relay(in_socket, out_socket, side_socket, msg, side_msg, id_msg, swap_ids)
            if rc < 0:
                return rc
        if items[1].revents & ZMQ_POLLIN:
            rc = zmq_msg_copy(address(side_msg), out_msg_ptr)
            if rc < 0:
                return rc
            rc = zmq_msg_send(address(side_msg), side_socket, ZMQ_SNDMORE)
            if rc < 0:
                return rc
            rc = _mq_relay(out_socket, in_socket, side_socket, msg, side_msg, id_msg, swap_ids)
            if rc < 0:
                return rc
    return rc