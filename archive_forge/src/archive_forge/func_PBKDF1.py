import re
import struct
from functools import reduce
from Cryptodome.Util.py3compat import (tobytes, bord, _copy_bytes, iter_range,
from Cryptodome.Hash import SHA1, SHA256, HMAC, CMAC, BLAKE2s
from Cryptodome.Util.strxor import strxor
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.number import size as bit_size, long_to_bytes, bytes_to_long
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
def PBKDF1(password, salt, dkLen, count=1000, hashAlgo=None):
    """Derive one key from a password (or passphrase).

    This function performs key derivation according to an old version of
    the PKCS#5 standard (v1.5) or `RFC2898
    <https://www.ietf.org/rfc/rfc2898.txt>`_.

    Args:
     password (string):
        The secret password to generate the key from.
     salt (byte string):
        An 8 byte string to use for better protection from dictionary attacks.
        This value does not need to be kept secret, but it should be randomly
        chosen for each derivation.
     dkLen (integer):
        The length of the desired key. The default is 16 bytes, suitable for
        instance for :mod:`Cryptodome.Cipher.AES`.
     count (integer):
        The number of iterations to carry out. The recommendation is 1000 or
        more.
     hashAlgo (module):
        The hash algorithm to use, as a module or an object from the :mod:`Cryptodome.Hash` package.
        The digest length must be no shorter than ``dkLen``.
        The default algorithm is :mod:`Cryptodome.Hash.SHA1`.

    Return:
        A byte string of length ``dkLen`` that can be used as key.
    """
    if not hashAlgo:
        hashAlgo = SHA1
    password = tobytes(password)
    pHash = hashAlgo.new(password + salt)
    digest = pHash.digest_size
    if dkLen > digest:
        raise TypeError('Selected hash algorithm has a too short digest (%d bytes).' % digest)
    if len(salt) != 8:
        raise ValueError('Salt is not 8 bytes long (%d bytes instead).' % len(salt))
    for i in iter_range(count - 1):
        pHash = pHash.new(pHash.digest())
    return pHash.digest()[:dkLen]