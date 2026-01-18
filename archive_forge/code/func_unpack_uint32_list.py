from binascii import hexlify
import hashlib
import logging; log = logging.getLogger(__name__)
import struct
import warnings
from passlib import exc
from passlib.utils import getrandbytes
from passlib.utils.compat import PYPY, u, bascii_to_str
from passlib.utils.decor import classproperty
from passlib.tests.utils import TestCase, skipUnless, TEST_MODE, hb
from passlib.crypto import scrypt as scrypt_mod
def unpack_uint32_list(data, check_count=None):
    """unpack bytes as list of uint32 values"""
    count = len(data) // 4
    assert check_count is None or check_count == count
    return struct.unpack('<%dI' % count, data)