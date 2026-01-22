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
@classmethod
def _parse_mlsx(cls, lines):
    for line in lines:
        name, facts = cls._parse_facts(line.strip())
        if name is None:
            continue
        _type = facts.get('type', 'file')
        if _type not in {'dir', 'file'}:
            continue
        is_dir = _type == 'dir'
        raw_info = {}
        raw_info['basic'] = {'name': name, 'is_dir': is_dir}
        raw_info['ftp'] = facts
        raw_info['details'] = {'type': int(ResourceType.directory if is_dir else ResourceType.file)}
        details = raw_info['details']
        size_str = facts.get('size', facts.get('sizd', '0'))
        size = 0
        if size_str.isdigit():
            size = int(size_str)
        details['size'] = size
        if 'modify' in facts:
            details['modified'] = cls._parse_ftp_time(facts['modify'])
        if 'create' in facts:
            details['created'] = cls._parse_ftp_time(facts['create'])
        yield raw_info