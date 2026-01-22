from __future__ import annotations
import binascii
import hashlib
def _hash_murmurhash(buf):
    """
        Produce a 16-bytes hash of *buf* using MurmurHash.
        """
    return mmh3.hash_bytes(buf)