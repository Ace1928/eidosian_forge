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
def _strip_last_newline(value):
    """Strip the last newline from value."""
    if value and value.endswith(b'\n'):
        return value[:-1]
    return value