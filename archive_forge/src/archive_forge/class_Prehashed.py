from __future__ import annotations
from cryptography.hazmat.bindings._rust import asn1
from cryptography.hazmat.primitives import hashes
class Prehashed:

    def __init__(self, algorithm: hashes.HashAlgorithm):
        if not isinstance(algorithm, hashes.HashAlgorithm):
            raise TypeError('Expected instance of HashAlgorithm.')
        self._algorithm = algorithm
        self._digest_size = algorithm.digest_size

    @property
    def digest_size(self) -> int:
        return self._digest_size