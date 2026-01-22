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
class _Queue(Generic[_T]):

    def __init__(self, incoming_packets_buffer: int | float):
        self.s, self.r = trio.open_memory_channel[_T](incoming_packets_buffer)