from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
class BLAKE2b(HashAlgorithm):
    name = 'blake2b'
    _max_digest_size = 64
    _min_digest_size = 1
    block_size = 128

    def __init__(self, digest_size: int):
        if digest_size != 64:
            raise ValueError('Digest size must be 64')
        self._digest_size = digest_size

    @property
    def digest_size(self) -> int:
        return self._digest_size