import os
import stat
import sys
import time
import warnings
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
from .hooks import (
from .line_ending import BlobNormalizer, TreeBlobNormalizer
from .object_store import (
from .objects import (
from .pack import generate_unpacked_objects
from .refs import (
def _set_filesystem_hidden(path):
    """Mark path as to be hidden if supported by platform and filesystem.

    On win32 uses SetFileAttributesW api:
    <https://docs.microsoft.com/windows/desktop/api/fileapi/nf-fileapi-setfileattributesw>
    """
    if sys.platform == 'win32':
        import ctypes
        from ctypes.wintypes import BOOL, DWORD, LPCWSTR
        FILE_ATTRIBUTE_HIDDEN = 2
        SetFileAttributesW = ctypes.WINFUNCTYPE(BOOL, LPCWSTR, DWORD)(('SetFileAttributesW', ctypes.windll.kernel32))
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        if not SetFileAttributesW(path, FILE_ATTRIBUTE_HIDDEN):
            pass