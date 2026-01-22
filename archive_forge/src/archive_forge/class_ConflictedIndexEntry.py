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
class ConflictedIndexEntry:
    """Index entry that represents a conflict."""
    ancestor: Optional[IndexEntry]
    this: Optional[IndexEntry]
    other: Optional[IndexEntry]

    def __init__(self, ancestor: Optional[IndexEntry]=None, this: Optional[IndexEntry]=None, other: Optional[IndexEntry]=None) -> None:
        self.ancestor = ancestor
        self.this = this
        self.other = other