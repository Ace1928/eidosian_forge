from Cryptodome.Util._raw_api import c_size_t
from Cryptodome.Hash.cSHAKE128 import cSHAKE_XOF
Return a fresh instance of a cSHAKE256 object.

    Args:
       data (bytes/bytearray/memoryview):
        The very first chunk of the message to hash.
        It is equivalent to an early call to :meth:`update`.
        Optional.
       custom (bytes):
        Optional.
        A customization bytestring (``S`` in SP 800-185).

    :Return: A :class:`cSHAKE_XOF` object
    