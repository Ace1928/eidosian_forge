import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import safe_crypt, test_crypt, to_unicode
from passlib.utils.binary import h64, h64big
from passlib.utils.compat import byte_elem_value, u, uascii_to_str, unicode, suppress_cause
from passlib.crypto.des import des_encrypt_int_block
import passlib.utils.handlers as uh
def _raw_des_crypt(secret, salt):
    """pure-python backed for des_crypt"""
    assert len(salt) == 2
    salt_value = h64.decode_int12(salt)
    if isinstance(secret, unicode):
        secret = secret.encode('utf-8')
    assert isinstance(secret, bytes)
    if _BNULL in secret:
        raise uh.exc.NullPasswordError(des_crypt)
    key_value = _crypt_secret_to_key(secret)
    result = des_encrypt_int_block(key_value, 0, salt_value, 25)
    return h64big.encode_int64(result)