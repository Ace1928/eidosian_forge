from __future__ import annotations
import socket
from abc import abstractmethod
from collections.abc import Callable, Collection, Mapping
from contextlib import AsyncExitStack
from io import IOBase
from ipaddress import IPv4Address, IPv6Address
from socket import AddressFamily
from types import TracebackType
from typing import Any, Tuple, TypeVar, Union
from .._core._typedattr import (
from ._streams import ByteStream, Listener, UnreliableObjectStream
from ._tasks import TaskGroup
class _SocketProvider(TypedAttributeProvider):

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        from .._core._sockets import convert_ipv6_sockaddr as convert
        attributes: dict[Any, Callable[[], Any]] = {SocketAttribute.family: lambda: self._raw_socket.family, SocketAttribute.local_address: lambda: convert(self._raw_socket.getsockname()), SocketAttribute.raw_socket: lambda: self._raw_socket}
        try:
            peername: tuple[str, int] | None = convert(self._raw_socket.getpeername())
        except OSError:
            peername = None
        if peername is not None:
            attributes[SocketAttribute.remote_address] = lambda: peername
        if self._raw_socket.family in (AddressFamily.AF_INET, AddressFamily.AF_INET6):
            attributes[SocketAttribute.local_port] = lambda: self._raw_socket.getsockname()[1]
            if peername is not None:
                remote_port = peername[1]
                attributes[SocketAttribute.remote_port] = lambda: remote_port
        return attributes

    @property
    @abstractmethod
    def _raw_socket(self) -> socket.socket:
        pass