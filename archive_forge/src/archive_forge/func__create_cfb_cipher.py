from Cryptodome.Util.py3compat import _copy_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
from Cryptodome.Random import get_random_bytes
def _create_cfb_cipher(factory, **kwargs):
    """Instantiate a cipher object that performs CFB encryption/decryption.

    :Parameters:
      factory : module
        The underlying block cipher, a module from ``Cryptodome.Cipher``.

    :Keywords:
      iv : bytes/bytearray/memoryview
        The IV to use for CFB.

      IV : bytes/bytearray/memoryview
        Alias for ``iv``.

      segment_size : integer
        The number of bit the plaintext and ciphertext are segmented in.
        If not present, the default is 8.

    Any other keyword will be passed to the underlying block cipher.
    See the relevant documentation for details (at least ``key`` will need
    to be present).
    """
    cipher_state = factory._create_base_cipher(kwargs)
    iv = kwargs.pop('IV', None)
    IV = kwargs.pop('iv', None)
    if (None, None) == (iv, IV):
        iv = get_random_bytes(factory.block_size)
    if iv is not None:
        if IV is not None:
            raise TypeError("You must either use 'iv' or 'IV', not both")
    else:
        iv = IV
    if len(iv) != factory.block_size:
        raise ValueError('Incorrect IV length (it must be %d bytes long)' % factory.block_size)
    segment_size_bytes, rem = divmod(kwargs.pop('segment_size', 8), 8)
    if segment_size_bytes == 0 or rem != 0:
        raise ValueError("'segment_size' must be positive and multiple of 8 bits")
    if kwargs:
        raise TypeError('Unknown parameters for CFB: %s' % str(kwargs))
    return CfbMode(cipher_state, iv, segment_size_bytes)