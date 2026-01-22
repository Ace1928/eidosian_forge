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
class FixedSha:
    """SHA object that behaves like hashlib's but is given a fixed value."""
    __slots__ = ('_hexsha', '_sha')

    def __init__(self, hexsha) -> None:
        if getattr(hexsha, 'encode', None) is not None:
            hexsha = hexsha.encode('ascii')
        if not isinstance(hexsha, bytes):
            raise TypeError('Expected bytes for hexsha, got %r' % hexsha)
        self._hexsha = hexsha
        self._sha = hex_to_sha(hexsha)

    def digest(self) -> bytes:
        """Return the raw SHA digest."""
        return self._sha

    def hexdigest(self) -> str:
        """Return the hex SHA digest."""
        return self._hexsha.decode('ascii')