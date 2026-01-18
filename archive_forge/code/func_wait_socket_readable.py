from __future__ import annotations
import errno
import os
import socket
import ssl
import stat
import sys
from collections.abc import Awaitable
from ipaddress import IPv6Address, ip_address
from os import PathLike, chmod
from socket import AddressFamily, SocketKind
from typing import Any, Literal, cast, overload
from .. import to_thread
from ..abc import (
from ..streams.stapled import MultiListener
from ..streams.tls import TLSStream
from ._eventloop import get_async_backend
from ._resources import aclose_forcefully
from ._synchronization import Event
from ._tasks import create_task_group, move_on_after
def wait_socket_readable(sock: socket.socket) -> Awaitable[None]:
    """
    Wait until the given socket has data to be read.

    This does **NOT** work on Windows when using the asyncio backend with a proactor
    event loop (default on py3.8+).

    .. warning:: Only use this on raw sockets that have not been wrapped by any higher
        level constructs like socket streams!

    :param sock: a socket object
    :raises ~anyio.ClosedResourceError: if the socket was closed while waiting for the
        socket to become readable
    :raises ~anyio.BusyResourceError: if another task is already waiting for the socket
        to become readable

    """
    return get_async_backend().wait_socket_readable(sock)