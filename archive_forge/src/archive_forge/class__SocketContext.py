from __future__ import annotations
import errno
import pickle
import random
import sys
from typing import (
from warnings import warn
import zmq
from zmq._typing import Literal, TypeAlias
from zmq.backend import Socket as SocketBase
from zmq.error import ZMQBindError, ZMQError
from zmq.utils import jsonapi
from zmq.utils.interop import cast_int_addr
from ..constants import SocketOption, SocketType, _OptType
from .attrsettr import AttributeSetter
from .poll import Poller
class _SocketContext(Generic[_SocketType]):
    """Context Manager for socket bind/unbind"""
    socket: _SocketType
    kind: str
    addr: str

    def __repr__(self):
        return f'<SocketContext({self.kind}={self.addr!r})>'

    def __init__(self: _SocketContext[_SocketType], socket: _SocketType, kind: str, addr: str):
        assert kind in {'bind', 'connect'}
        self.socket = socket
        self.kind = kind
        self.addr = addr

    def __enter__(self: _SocketContext[_SocketType]) -> _SocketType:
        return self.socket

    def __exit__(self, *args):
        if self.socket.closed:
            return
        if self.kind == 'bind':
            self.socket.unbind(self.addr)
        elif self.kind == 'connect':
            self.socket.disconnect(self.addr)