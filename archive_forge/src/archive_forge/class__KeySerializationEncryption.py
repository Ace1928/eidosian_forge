from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.hazmat.primitives.hashes import HashAlgorithm
class _KeySerializationEncryption(KeySerializationEncryption):

    def __init__(self, format: PrivateFormat, password: bytes, *, kdf_rounds: typing.Optional[int], hmac_hash: typing.Optional[HashAlgorithm], key_cert_algorithm: typing.Optional[PBES]):
        self._format = format
        self.password = password
        self._kdf_rounds = kdf_rounds
        self._hmac_hash = hmac_hash
        self._key_cert_algorithm = key_cert_algorithm