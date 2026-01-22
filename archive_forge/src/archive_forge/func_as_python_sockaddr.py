from __future__ import annotations
import contextlib
import errno
import ipaddress
import os
import socket
import sys
from typing import (
import attrs
import trio
from trio._util import NoPublicConstructor, final
def as_python_sockaddr(self) -> tuple[str, int] | tuple[str, int, int, int]:
    sockaddr: tuple[str, int] | tuple[str, int, int, int] = (self.ip.compressed, self.port)
    if isinstance(self.ip, ipaddress.IPv6Address):
        sockaddr += (0, 0)
    return sockaddr