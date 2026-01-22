from __future__ import print_function, unicode_literals
import typing
import array
import calendar
import datetime
import io
import itertools
import socket
import threading
from collections import OrderedDict
from contextlib import contextmanager
from ftplib import FTP
from typing import cast
from ftplib import error_perm, error_temp
from six import PY2, raise_from, text_type
from . import _ftp_parse as ftp_parse
from . import errors
from .base import FS
from .constants import DEFAULT_CHUNK_SIZE
from .enums import ResourceType, Seek
from .info import Info
from .iotools import line_iterator
from .mode import Mode
from .path import abspath, basename, dirname, normpath, split
from .time import epoch_to_datetime
def _read_dir(self, path):
    _path = abspath(normpath(path))
    lines = []
    with ftp_errors(self, path=path):
        self.ftp.retrlines(str('LIST ') + _encode(_path, self.ftp.encoding), lines.append)
    lines = [line.decode('utf-8') if isinstance(line, bytes) else line for line in lines]
    _list = [Info(raw_info) for raw_info in ftp_parse.parse(lines)]
    dir_listing = OrderedDict({info.name: info for info in _list})
    return dir_listing