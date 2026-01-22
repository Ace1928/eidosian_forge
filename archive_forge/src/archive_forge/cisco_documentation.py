from binascii import hexlify, unhexlify
from hashlib import md5
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import right_pad_string, to_unicode, repeat_string, to_bytes
from passlib.utils.binary import h64
from passlib.utils.compat import unicode, u, join_byte_values, \
import passlib.utils.handlers as uh
xor static key against data - encrypts & decrypts