from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.hazmat._oid import ObjectIdentifier
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
class ECDSA(EllipticCurveSignatureAlgorithm):

    def __init__(self, algorithm: typing.Union[asym_utils.Prehashed, hashes.HashAlgorithm]):
        self._algorithm = algorithm

    @property
    def algorithm(self) -> typing.Union[asym_utils.Prehashed, hashes.HashAlgorithm]:
        return self._algorithm