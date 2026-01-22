import collections.abc
import contextlib
import datetime
import errno
import functools
import io
import os
import pathlib
import queue
import re
import stat
import sys
import time
from multiprocessing import Process
from threading import Thread
from typing import IO, Any, BinaryIO, Collection, Dict, List, Optional, Tuple, Type, Union
import multivolumefile
from py7zr.archiveinfo import Folder, Header, SignatureHeader
from py7zr.callbacks import ExtractCallback
from py7zr.compressor import SupportedMethods, get_methods_names
from py7zr.exceptions import (
from py7zr.helpers import (
from py7zr.properties import DEFAULT_FILTERS, FILTER_DEFLATE64, MAGIC_7Z, get_default_blocksize, get_memory_limit
@staticmethod
def _make_file_info(target: pathlib.Path, arcname: Optional[str]=None, dereference=False) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    f['origin'] = target
    if arcname is not None:
        f['filename'] = pathlib.Path(arcname).as_posix()
    else:
        f['filename'] = target.as_posix()
    if sys.platform == 'win32':
        fstat = target.lstat()
        if target.is_symlink():
            if dereference:
                fstat = target.stat()
                if stat.S_ISDIR(fstat.st_mode):
                    f['emptystream'] = True
                    f['attributes'] = fstat.st_file_attributes & FILE_ATTRIBUTE_WINDOWS_MASK
                else:
                    f['emptystream'] = False
                    f['attributes'] = stat.FILE_ATTRIBUTE_ARCHIVE
                    f['uncompressed'] = fstat.st_size
            else:
                f['emptystream'] = False
                f['attributes'] = fstat.st_file_attributes & FILE_ATTRIBUTE_WINDOWS_MASK
        elif target.is_dir():
            f['emptystream'] = True
            f['attributes'] = fstat.st_file_attributes & FILE_ATTRIBUTE_WINDOWS_MASK
        elif target.is_file():
            f['emptystream'] = False
            f['attributes'] = stat.FILE_ATTRIBUTE_ARCHIVE
            f['uncompressed'] = fstat.st_size
    elif sys.platform == 'darwin' or sys.platform.startswith('linux') or sys.platform.startswith('freebsd') or sys.platform.startswith('netbsd') or sys.platform.startswith('sunos') or (sys.platform == 'aix'):
        fstat = target.lstat()
        if target.is_symlink():
            if dereference:
                fstat = target.stat()
                if stat.S_ISDIR(fstat.st_mode):
                    f['emptystream'] = True
                    f['attributes'] = getattr(stat, 'FILE_ATTRIBUTE_DIRECTORY')
                    f['attributes'] |= FILE_ATTRIBUTE_UNIX_EXTENSION | stat.S_IFDIR << 16
                    f['attributes'] |= stat.S_IMODE(fstat.st_mode) << 16
                else:
                    f['emptystream'] = False
                    f['attributes'] = getattr(stat, 'FILE_ATTRIBUTE_ARCHIVE')
                    f['attributes'] |= FILE_ATTRIBUTE_UNIX_EXTENSION | stat.S_IMODE(fstat.st_mode) << 16
            else:
                f['emptystream'] = False
                f['attributes'] = getattr(stat, 'FILE_ATTRIBUTE_ARCHIVE') | getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT')
                f['attributes'] |= FILE_ATTRIBUTE_UNIX_EXTENSION | stat.S_IFLNK << 16
                f['attributes'] |= stat.S_IMODE(fstat.st_mode) << 16
        elif target.is_dir():
            f['emptystream'] = True
            f['attributes'] = getattr(stat, 'FILE_ATTRIBUTE_DIRECTORY')
            f['attributes'] |= FILE_ATTRIBUTE_UNIX_EXTENSION | stat.S_IFDIR << 16
            f['attributes'] |= stat.S_IMODE(fstat.st_mode) << 16
        elif target.is_file():
            f['emptystream'] = False
            f['uncompressed'] = fstat.st_size
            f['attributes'] = getattr(stat, 'FILE_ATTRIBUTE_ARCHIVE')
            f['attributes'] |= FILE_ATTRIBUTE_UNIX_EXTENSION | stat.S_IMODE(fstat.st_mode) << 16
    else:
        fstat = target.stat()
        if target.is_dir():
            f['emptystream'] = True
            f['attributes'] = stat.FILE_ATTRIBUTE_DIRECTORY
        elif target.is_file():
            f['emptystream'] = False
            f['uncompressed'] = fstat.st_size
            f['attributes'] = stat.FILE_ATTRIBUTE_ARCHIVE
    f['creationtime'] = ArchiveTimestamp.from_datetime(fstat.st_ctime)
    f['lastwritetime'] = ArchiveTimestamp.from_datetime(fstat.st_mtime)
    f['lastaccesstime'] = ArchiveTimestamp.from_datetime(fstat.st_atime)
    return f