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
def validate_path_element_ntfs(element: bytes) -> bool:
    stripped = element.rstrip(b'. ').lower()
    if stripped in INVALID_DOTNAMES:
        return False
    if stripped == b'git~1':
        return False
    return True