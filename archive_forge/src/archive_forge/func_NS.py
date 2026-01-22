import struct
from cryptography.utils import int_to_bytes
from twisted.python.deprecate import deprecated
from twisted.python.versions import Version
def NS(t):
    """
    net string
    """
    if isinstance(t, str):
        t = t.encode('utf-8')
    return struct.pack('!L', len(t)) + t