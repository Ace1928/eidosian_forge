import nacl.exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
Compute a fast, cryptographic quality, keyed hash of the input data

    :param data:
    :type data: bytes
    :param key: len(key) must be equal to
                :py:data:`.XKEYBYTES` (16)
    :type key: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    