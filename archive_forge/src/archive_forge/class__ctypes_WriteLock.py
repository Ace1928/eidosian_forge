import contextlib
import errno
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple
from . import debug, errors, osutils, trace
from .hooks import Hooks
from .i18n import gettext
from .transport import Transport
class _ctypes_WriteLock(_ctypes_FileLock):

    def __init__(self, filename):
        super().__init__()
        self._open(filename, GENERIC_READ | GENERIC_WRITE, 0, os.O_RDWR, 'rb+')

    def restore_read_lock(self):
        """Restore the original ReadLock."""
        self.unlock()
        return _ctypes_ReadLock(self.filename)