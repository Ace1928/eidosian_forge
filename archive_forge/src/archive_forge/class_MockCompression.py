import binascii
import re
import string
import struct
import types
from hashlib import md5, sha1, sha256, sha384, sha512
from typing import Dict, List, Optional, Tuple, Type
from twisted import __version__ as twisted_version
from twisted.conch.error import ConchError
from twisted.conch.ssh import _kex, address, service
from twisted.internet import defer
from twisted.protocols import loopback
from twisted.python import randbytes
from twisted.python.compat import iterbytes
from twisted.python.randbytes import insecureRandom
from twisted.python.reflect import requireModule
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
class MockCompression:
    """
    A mocked-up compression, based on the zlib interface.  Instead of
    compressing, it reverses the data and adds a 0x66 byte to the end.
    """

    def compress(self, payload):
        return payload[::-1]

    def decompress(self, payload):
        return payload[:-1][::-1]

    def flush(self, kind):
        return b'f'