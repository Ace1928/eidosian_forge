import errno
import io
import os
import sys
import socket
import struct
import time
import tempfile
import itertools
from . import util
from . import AuthenticationError, BufferTooShort
from .context import reduction
def _get_digest_name_and_payload(message: bytes) -> (str, bytes):
    """Returns a digest name and the payload for a response hash.

    If a legacy protocol is detected based on the message length
    or contents the digest name returned will be empty to indicate
    legacy mode where MD5 and no digest prefix should be sent.
    """
    if len(message) in _LEGACY_LENGTHS:
        return ('', message)
    if message.startswith(b'{') and (curly := message.find(b'}', 1, _MAX_DIGEST_LEN + 2)) > 0:
        digest = message[1:curly]
        if digest in _ALLOWED_DIGESTS:
            payload = message[curly + 1:]
            return (digest.decode('ascii'), payload)
    raise AuthenticationError(f'unsupported message length, missing digest prefix, or unsupported digest: message={message!r}')