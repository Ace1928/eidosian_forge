import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import safe_crypt, test_crypt, to_unicode
from passlib.utils.binary import h64, h64big
from passlib.utils.compat import byte_elem_value, u, uascii_to_str, unicode, suppress_cause
from passlib.crypto.des import des_encrypt_int_block
import passlib.utils.handlers as uh
def _bsdi_secret_to_key(secret):
    """convert secret to DES key used by bsdi_crypt"""
    key_value = _crypt_secret_to_key(secret)
    idx = 8
    end = len(secret)
    while idx < end:
        next = idx + 8
        tmp_value = _crypt_secret_to_key(secret[idx:next])
        key_value = des_encrypt_int_block(key_value, key_value) ^ tmp_value
        idx = next
    return key_value