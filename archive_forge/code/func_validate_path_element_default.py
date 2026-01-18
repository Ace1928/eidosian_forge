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
def validate_path_element_default(element: bytes) -> bool:
    return element.lower() not in INVALID_DOTNAMES