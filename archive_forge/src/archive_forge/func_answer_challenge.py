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
def answer_challenge(connection, authkey):
    import hmac
    if not isinstance(authkey, bytes):
        raise ValueError('Authkey must be bytes, not {0!s}'.format(type(authkey)))
    message = connection.recv_bytes(256)
    assert message[:len(CHALLENGE)] == CHALLENGE, 'message = %r' % message
    message = message[len(CHALLENGE):]
    digest = hmac.new(authkey, message, 'md5').digest()
    connection.send_bytes(digest)
    response = connection.recv_bytes(256)
    if response != WELCOME:
        raise AuthenticationError('digest sent was rejected')