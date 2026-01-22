import contextlib
import ctypes
import errno
import logging
import os
import platform
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import IO, Any, BinaryIO, Generator, Optional
from wandb.sdk.lib.paths import StrPath
def _reflink_macos(existing_path: Path, new_path: Path) -> None:
    try:
        clib = ctypes.CDLL('libc.dylib', use_errno=True)
    except (FileNotFoundError, OSError) as e:
        if ctypes.get_errno() != errno.ENOENT and (not isinstance(e, FileNotFoundError)):
            raise
        clib = ctypes.CDLL('/usr/lib/libSystem.dylib', use_errno=True)
    try:
        clonefile = clib.clonefile
    except AttributeError:
        raise OSError(errno.ENOTSUP, "'clonefile' is not available on this system")
    clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
    clonefile.restype = ctypes.c_int
    if clonefile(os.fsencode(existing_path), os.fsencode(new_path), ctypes.c_int(0)):
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), existing_path)