import struct
from binascii import unhexlify
from Cryptodome.Util.py3compat import byte_string, bord, _copy_bytes
from Cryptodome.Util._raw_api import is_buffer
from Cryptodome.Util.strxor import strxor
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Hash import CMAC, BLAKE2s
from Cryptodome.Random import get_random_bytes
def _create_eax_cipher(factory, **kwargs):
    """Create a new block cipher, configured in EAX mode.

    :Parameters:
      factory : module
        A symmetric cipher module from `Cryptodome.Cipher` (like
        `Cryptodome.Cipher.AES`).

    :Keywords:
      key : bytes/bytearray/memoryview
        The secret key to use in the symmetric cipher.

      nonce : bytes/bytearray/memoryview
        A value that must never be reused for any other encryption.
        There are no restrictions on its length, but it is recommended to use
        at least 16 bytes.

        The nonce shall never repeat for two different messages encrypted with
        the same key, but it does not need to be random.

        If not specified, a 16 byte long random string is used.

      mac_len : integer
        Length of the MAC, in bytes. It must be no larger than the cipher
        block bytes (which is the default).
    """
    try:
        key = kwargs.pop('key')
        nonce = kwargs.pop('nonce', None)
        if nonce is None:
            nonce = get_random_bytes(16)
        mac_len = kwargs.pop('mac_len', factory.block_size)
    except KeyError as e:
        raise TypeError('Missing parameter: ' + str(e))
    return EaxMode(factory, key, nonce, mac_len, kwargs)