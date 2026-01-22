from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import bord, _copy_bytes
from Cryptodome.Util._raw_api import is_buffer
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Protocol.KDF import _S2V
from Cryptodome.Hash import BLAKE2s
from Cryptodome.Random import get_random_bytes
def _create_ctr_cipher(self, v):
    """Create a new CTR cipher from V in SIV mode"""
    v_int = bytes_to_long(v)
    q = v_int & 340282366920938463454151235392765951999
    return self._factory.new(self._subkey_cipher, self._factory.MODE_CTR, initial_value=q, nonce=b'', **self._cipher_params)