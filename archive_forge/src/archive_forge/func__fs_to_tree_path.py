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
def _fs_to_tree_path(fs_path: Union[str, bytes]) -> bytes:
    """Convert a file system path to a git tree path.

    Args:
      fs_path: File system path.

    Returns:  Git tree path as bytes
    """
    if not isinstance(fs_path, bytes):
        fs_path_bytes = os.fsencode(fs_path)
    else:
        fs_path_bytes = fs_path
    if os_sep_bytes != b'/':
        tree_path = fs_path_bytes.replace(os_sep_bytes, b'/')
    else:
        tree_path = fs_path_bytes
    return tree_path