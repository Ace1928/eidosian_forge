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
def _signable(*fields: bytes) -> bytes:
    out = []
    for field in fields:
        out.append(struct.pack('!Q', len(field)))
        out.append(field)
    return b''.join(out)