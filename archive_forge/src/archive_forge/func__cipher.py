from binascii import hexlify, unhexlify
from hashlib import md5
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import right_pad_string, to_unicode, repeat_string, to_bytes
from passlib.utils.binary import h64
from passlib.utils.compat import unicode, u, join_byte_values, \
import passlib.utils.handlers as uh
@classmethod
def _cipher(cls, data, salt):
    """xor static key against data - encrypts & decrypts"""
    key = cls._key
    key_size = len(key)
    return join_byte_values((value ^ ord(key[(salt + idx) % key_size]) for idx, value in enumerate(iter_byte_values(data))))