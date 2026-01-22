import struct
from binascii import unhexlify
from Cryptodome.Util.py3compat import (byte_string, bord,
from Cryptodome.Util._raw_api import is_writeable_buffer
from Cryptodome.Util.strxor import strxor
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Hash import BLAKE2s
from Cryptodome.Random import get_random_bytes
def _start_mac(self):
    assert self._mac_status == MacStatus.NOT_STARTED
    assert None not in (self._assoc_len, self._msg_len)
    assert isinstance(self._cache, list)
    q = 15 - len(self.nonce)
    flags = 64 * (self._assoc_len > 0) + 8 * ((self._mac_len - 2) // 2) + (q - 1)
    b_0 = struct.pack('B', flags) + self.nonce + long_to_bytes(self._msg_len, q)
    assoc_len_encoded = b''
    if self._assoc_len > 0:
        if self._assoc_len < 2 ** 16 - 2 ** 8:
            enc_size = 2
        elif self._assoc_len < 2 ** 32:
            assoc_len_encoded = b'\xff\xfe'
            enc_size = 4
        else:
            assoc_len_encoded = b'\xff\xff'
            enc_size = 8
        assoc_len_encoded += long_to_bytes(self._assoc_len, enc_size)
    self._cache.insert(0, b_0)
    self._cache.insert(1, assoc_len_encoded)
    first_data_to_mac = b''.join(self._cache)
    self._cache = b''
    self._mac_status = MacStatus.PROCESSING_AUTH_DATA
    self._update(first_data_to_mac)