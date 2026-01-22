from __future__ import annotations
import hashlib
import os
from textwrap import dedent
from typing import IO, TYPE_CHECKING
from pip._vendor.cachecontrol.cache import BaseCache, SeparateBodyBaseCache
from pip._vendor.cachecontrol.controller import CacheController
def _secure_open_write(filename: str, fmode: int) -> IO[bytes]:
    flags = os.O_WRONLY
    flags |= os.O_CREAT | os.O_EXCL
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    if hasattr(os, 'O_BINARY'):
        flags |= os.O_BINARY
    try:
        os.remove(filename)
    except OSError:
        pass
    fd = os.open(filename, flags, fmode)
    try:
        return os.fdopen(fd, 'wb')
    except:
        os.close(fd)
        raise