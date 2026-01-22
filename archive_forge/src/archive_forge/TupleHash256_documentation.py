from . import cSHAKE256
from .TupleHash128 import TupleHash
Create a new TupleHash256 object.

    Args:
       digest_bytes (integer):
        Optional. The size of the digest, in bytes.
        Default is 64. Minimum is 8.
       digest_bits (integer):
        Optional and alternative to ``digest_bytes``.
        The size of the digest, in bits (and in steps of 8).
        Default is 512. Minimum is 64.
       custom (bytes):
        Optional.
        A customization bytestring (``S`` in SP 800-185).

    :Return: A :class:`TupleHash` object
    