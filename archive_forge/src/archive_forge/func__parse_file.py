import binascii
import os
import posixpath
import stat
import warnings
import zlib
from collections import namedtuple
from hashlib import sha1
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
@classmethod
def _parse_file(cls, f):
    map = f.read()
    if not map:
        raise EmptyFileException('Corrupted empty file detected')
    if cls._is_legacy_object(map):
        obj = cls._parse_legacy_object_header(map, f)
        obj._parse_legacy_object(map)
    else:
        obj = cls._parse_object_header(map, f)
        obj._parse_object(map)
    return obj