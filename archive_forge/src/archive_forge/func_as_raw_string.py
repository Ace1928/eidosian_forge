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
def as_raw_string(self) -> bytes:
    """Return raw string with serialization of the object.

        Returns: String object
        """
    return b''.join(self.as_raw_chunks())