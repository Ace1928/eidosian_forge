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
@dataclass
class SerializedIndexEntry:
    name: bytes
    ctime: Union[int, float, Tuple[int, int]]
    mtime: Union[int, float, Tuple[int, int]]
    dev: int
    ino: int
    mode: int
    uid: int
    gid: int
    size: int
    sha: bytes
    flags: int
    extended_flags: int

    def stage(self) -> Stage:
        return Stage((self.flags & FLAG_STAGEMASK) >> FLAG_STAGESHIFT)