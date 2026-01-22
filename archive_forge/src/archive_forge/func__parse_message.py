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
def _parse_message(chunks: Iterable[bytes]) -> Iterator[Union[Tuple[None, None], Tuple[Optional[bytes], bytes]]]:
    """Parse a message with a list of fields and a body.

    Args:
      chunks: the raw chunks of the tag or commit object.
    Returns: iterator of tuples of (field, value), one per header line, in the
        order read from the text, possibly including duplicates. Includes a
        field named None for the freeform tag/commit text.
    """
    f = BytesIO(b''.join(chunks))
    k = None
    v = b''
    eof = False

    def _strip_last_newline(value):
        """Strip the last newline from value."""
        if value and value.endswith(b'\n'):
            return value[:-1]
        return value
    for line in f:
        if line.startswith(b' '):
            v += line[1:]
        else:
            if k is not None:
                yield (k, _strip_last_newline(v))
            if line == b'\n':
                break
            k, v = line.split(b' ', 1)
    else:
        eof = True
        if k is not None:
            yield (k, _strip_last_newline(v))
        yield (None, None)
    if not eof:
        yield (None, f.read())
    f.close()