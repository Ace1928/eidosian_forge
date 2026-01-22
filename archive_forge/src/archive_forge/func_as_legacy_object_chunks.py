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
def as_legacy_object_chunks(self, compression_level: int=-1) -> Iterator[bytes]:
    """Return chunks representing the object in the experimental format.

        Returns: List of strings
        """
    compobj = zlib.compressobj(compression_level)
    yield compobj.compress(self._header())
    for chunk in self.as_raw_chunks():
        yield compobj.compress(chunk)
    yield compobj.flush()