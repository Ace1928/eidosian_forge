from __future__ import annotations
import contextlib
import enum
import errno
import hmac
import os
import struct
import warnings
import weakref
from itertools import count
from typing import (
from weakref import ReferenceType, WeakValueDictionary
import attrs
import trio
from ._util import NoPublicConstructor, final
def worst_case_mtu(sock: SocketType) -> int:
    if sock.family == trio.socket.AF_INET:
        return 576 - packet_header_overhead(sock)
    else:
        return 1280 - packet_header_overhead(sock)