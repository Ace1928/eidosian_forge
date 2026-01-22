from __future__ import annotations
import errno
import inspect
import os
import socket as stdlib_socket
import sys
import tempfile
from socket import AddressFamily, SocketKind
from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union
import attrs
import pytest
from .. import _core, socket as tsocket
from .._core._tests.tutil import binds_ipv6, creates_ipv6
from .._socket import _NUMERIC_ONLY, SocketType, _SocketType, _try_sync
from ..testing import assert_checkpoints, wait_all_tasks_blocked
class CustomResolver:

    async def getaddrinfo(self, host: str, port: str, family: int, type: int, proto: int, flags: int) -> tuple[str, str, str, int, int, int, int]:
        return ('custom_gai', host, port, family, type, proto, flags)

    async def getnameinfo(self, sockaddr: tuple[str, int] | tuple[str, int, int, int], flags: int) -> tuple[str, tuple[str, int] | tuple[str, int, int, int], int]:
        return ('custom_gni', sockaddr, flags)