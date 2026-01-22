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
class WindowsSymlinkPermissionError(PermissionError):

    def __init__(self, errno, msg, filename) -> None:
        super(PermissionError, self).__init__(errno, 'Unable to create symlink; do you have developer mode enabled? %s' % msg, filename)