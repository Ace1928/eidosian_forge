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
class UnsupportedIndexFormat(Exception):
    """An unsupported index format was encountered."""

    def __init__(self, version) -> None:
        self.index_format_version = version