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
def _parse_legacy_object(self, map) -> None:
    """Parse a legacy object, setting the raw string."""
    text = _decompress(map)
    header_end = text.find(b'\x00')
    if header_end < 0:
        raise ObjectFormatException('Invalid object header, no \\0')
    self.set_raw_string(text[header_end + 1:])