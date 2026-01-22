import sys
from Cryptodome.Cipher import _create_cipher
from Cryptodome.Util.py3compat import byte_string, bchr, bord, bstr
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
def adjust_key_parity(key_in):
    """Set the parity bits in a TDES key.

    :param key_in: the TDES key whose bits need to be adjusted
    :type key_in: byte string

    :returns: a copy of ``key_in``, with the parity bits correctly set
    :rtype: byte string

    :raises ValueError: if the TDES key is not 16 or 24 bytes long
    :raises ValueError: if the TDES key degenerates into Single DES
    """

    def parity_byte(key_byte):
        parity = 1
        for i in range(1, 8):
            parity ^= key_byte >> i & 1
        return key_byte & 254 | parity
    if len(key_in) not in key_size:
        raise ValueError('Not a valid TDES key')
    key_out = b''.join([bchr(parity_byte(bord(x))) for x in key_in])
    if key_out[:8] == key_out[8:16] or key_out[-16:-8] == key_out[-8:]:
        raise ValueError('Triple DES key degenerates to single DES')
    return key_out