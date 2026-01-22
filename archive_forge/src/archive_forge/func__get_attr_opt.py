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
def _get_attr_opt(self, name: str, opt: int) -> OptValT:
    """get default sockopts as attributes"""
    if name in ContextOption.__members__:
        return self.get(opt)
    elif opt not in self.sockopts:
        raise AttributeError(name)
    else:
        return self.sockopts[opt]