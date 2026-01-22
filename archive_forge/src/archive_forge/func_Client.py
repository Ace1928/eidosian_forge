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
def Client(address, family=None, authkey=None):
    """
    Returns a connection to the address of a `Listener`
    """
    family = family or address_type(address)
    _validate_family(family)
    if family == 'AF_PIPE':
        c = PipeClient(address)
    else:
        c = SocketClient(address)
    if authkey is not None and (not isinstance(authkey, bytes)):
        raise TypeError('authkey should be a byte string')
    if authkey is not None:
        answer_challenge(c, authkey)
        deliver_challenge(c, authkey)
    return c