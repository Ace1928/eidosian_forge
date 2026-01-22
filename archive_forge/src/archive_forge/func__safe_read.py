from __future__ import annotations
import io
import itertools
import struct
import sys
from typing import Any, NamedTuple
from . import Image
from ._deprecate import deprecate
from ._util import is_path
def _safe_read(fp, size):
    """
    Reads large blocks in a safe way.  Unlike fp.read(n), this function
    doesn't trust the user.  If the requested size is larger than
    SAFEBLOCK, the file is read block by block.

    :param fp: File handle.  Must implement a <b>read</b> method.
    :param size: Number of bytes to read.
    :returns: A string containing <i>size</i> bytes of data.

    Raises an OSError if the file is truncated and the read cannot be completed

    """
    if size <= 0:
        return b''
    if size <= SAFEBLOCK:
        data = fp.read(size)
        if len(data) < size:
            msg = 'Truncated File Read'
            raise OSError(msg)
        return data
    data = []
    remaining_size = size
    while remaining_size > 0:
        block = fp.read(min(remaining_size, SAFEBLOCK))
        if not block:
            break
        data.append(block)
        remaining_size -= len(block)
    if sum((len(d) for d in data)) < size:
        msg = 'Truncated File Read'
        raise OSError(msg)
    return b''.join(data)