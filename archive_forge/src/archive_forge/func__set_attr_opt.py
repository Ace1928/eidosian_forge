from __future__ import annotations
import atexit
import os
from threading import Lock
from typing import Any, Callable, Generic, TypeVar, overload
from warnings import warn
from weakref import WeakSet
import zmq
from zmq._typing import TypeAlias
from zmq.backend import Context as ContextBase
from zmq.constants import ContextOption, Errno, SocketOption
from zmq.error import ZMQError
from zmq.utils.interop import cast_int_addr
from .attrsettr import AttributeSetter, OptValT
from .socket import Socket, SyncSocket
def _set_attr_opt(self, name: str, opt: int, value: OptValT) -> None:
    """set default sockopts as attributes"""
    if name in ContextOption.__members__:
        return self.set(opt, value)
    elif name in SocketOption.__members__:
        self.sockopts[opt] = value
    else:
        raise AttributeError(f'No such context or socket option: {name}')