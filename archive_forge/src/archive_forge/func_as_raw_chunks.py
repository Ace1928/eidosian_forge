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
def as_raw_chunks(self) -> List[bytes]:
    """Return chunks with serialization of the object.

        Returns: List of strings, not necessarily one per line
        """
    if self._needs_serialization:
        self._sha = None
        self._chunked_text = self._serialize()
        self._needs_serialization = False
    return self._chunked_text