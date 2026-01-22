from eventlet.patcher import slurp_properties
import sys
from eventlet import greenio, hubs
from eventlet.greenio import (
from eventlet.hubs import trampoline, IOClosed
from eventlet.support import get_errno, PY33
from contextlib import contextmanager
def _socket_connect(self, addr):
    real_connect = socket.connect
    if self.act_non_blocking:
        return real_connect(self, addr)
    else:
        clock = hubs.get_hub().clock
        if self.gettimeout() is None:
            while True:
                try:
                    return real_connect(self, addr)
                except orig_socket.error as exc:
                    if get_errno(exc) in CONNECT_ERR:
                        trampoline(self, write=True)
                    elif get_errno(exc) in CONNECT_SUCCESS:
                        return
                    else:
                        raise
        else:
            end = clock() + self.gettimeout()
            while True:
                try:
                    real_connect(self, addr)
                except orig_socket.error as exc:
                    if get_errno(exc) in CONNECT_ERR:
                        trampoline(self, write=True, timeout=end - clock(), timeout_exc=timeout_exc('timed out'))
                    elif get_errno(exc) in CONNECT_SUCCESS:
                        return
                    else:
                        raise
                if clock() >= end:
                    raise timeout_exc('timed out')