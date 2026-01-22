from binascii import unhexlify
from Cryptodome.Util.py3compat import bord, _copy_bytes
from Cryptodome.Util._raw_api import is_buffer
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Hash import BLAKE2s
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
from Cryptodome.Util import _cpu_features
def _create_gcm_cipher(factory, **kwargs):
    """Create a new block cipher, configured in Galois Counter Mode (GCM).

    :Parameters:
      factory : module
        A block cipher module, taken from `Cryptodome.Cipher`.
        The cipher must have block length of 16 bytes.
        GCM has been only defined for `Cryptodome.Cipher.AES`.

    :Keywords:
      key : bytes/bytearray/memoryview
        The secret key to use in the symmetric cipher.
        It must be 16 (e.g. *AES-128*), 24 (e.g. *AES-192*)
        or 32 (e.g. *AES-256*) bytes long.

      nonce : bytes/bytearray/memoryview
        A value that must never be reused for any other encryption.

        There are no restrictions on its length,
        but it is recommended to use at least 16 bytes.

        The nonce shall never repeat for two
        different messages encrypted with the same key,
        but it does not need to be random.

        If not provided, a 16 byte nonce will be randomly created.

      mac_len : integer
        Length of the MAC, in bytes.
        It must be no larger than 16 bytes (which is the default).
    """
    try:
        key = kwargs.pop('key')
    except KeyError as e:
        raise TypeError('Missing parameter:' + str(e))
    nonce = kwargs.pop('nonce', None)
    if nonce is None:
        nonce = get_random_bytes(16)
    mac_len = kwargs.pop('mac_len', 16)
    use_clmul = kwargs.pop('use_clmul', True)
    if use_clmul and _ghash_clmul:
        ghash_c = _ghash_clmul
    else:
        ghash_c = _ghash_portable
    return GcmMode(factory, key, nonce, mac_len, kwargs, ghash_c)