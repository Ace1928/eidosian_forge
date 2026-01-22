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
def _verify_challenge(authkey, message, response):
    """Verify MAC challenge

    If our message did not include a digest_name prefix, the client is allowed
    to select a stronger digest_name from _ALLOWED_DIGESTS.

    In case our message is prefixed, a client cannot downgrade to a weaker
    algorithm, because the MAC is calculated over the entire message
    including the '{digest_name}' prefix.
    """
    import hmac
    response_digest, response_mac = _get_digest_name_and_payload(response)
    response_digest = response_digest or 'md5'
    try:
        expected = hmac.new(authkey, message, response_digest).digest()
    except ValueError:
        raise AuthenticationError(f'response_digest={response_digest!r} unsupported')
    if len(expected) != len(response_mac):
        raise AuthenticationError(f'expected {response_digest!r} of length {len(expected)} got {len(response_mac)}')
    if not hmac.compare_digest(expected, response_mac):
        raise AuthenticationError('digest received was wrong')