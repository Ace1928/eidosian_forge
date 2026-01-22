from __future__ import annotations
import abc
import typing
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives._asymmetric import (
from cryptography.hazmat.primitives.asymmetric import rsa
class OAEP(AsymmetricPadding):
    name = 'EME-OAEP'

    def __init__(self, mgf: MGF, algorithm: hashes.HashAlgorithm, label: typing.Optional[bytes]):
        if not isinstance(algorithm, hashes.HashAlgorithm):
            raise TypeError('Expected instance of hashes.HashAlgorithm.')
        self._mgf = mgf
        self._algorithm = algorithm
        self._label = label