import os
import stat
import struct
import sys
from dataclasses import dataclass
from enum import Enum
from typing import (
from .file import GitFile
from .object_store import iter_tree_contents
from .objects import (
from .pack import ObjectContainer, SHA1Reader, SHA1Writer
def write_cache_time(f, t):
    """Write a cache time.

    Args:
      f: File-like object to write to
      t: Time to write (as int, float or tuple with secs and nsecs)
    """
    if isinstance(t, int):
        t = (t, 0)
    elif isinstance(t, float):
        secs, nsecs = divmod(t, 1.0)
        t = (int(secs), int(nsecs * 1000000000))
    elif not isinstance(t, tuple):
        raise TypeError(t)
    f.write(struct.pack('>LL', *t))