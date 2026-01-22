from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.hazmat.primitives.hashes import HashAlgorithm
class KeySerializationEncryptionBuilder:

    def __init__(self, format: PrivateFormat, *, _kdf_rounds: typing.Optional[int]=None, _hmac_hash: typing.Optional[HashAlgorithm]=None, _key_cert_algorithm: typing.Optional[PBES]=None) -> None:
        self._format = format
        self._kdf_rounds = _kdf_rounds
        self._hmac_hash = _hmac_hash
        self._key_cert_algorithm = _key_cert_algorithm

    def kdf_rounds(self, rounds: int) -> KeySerializationEncryptionBuilder:
        if self._kdf_rounds is not None:
            raise ValueError('kdf_rounds already set')
        if not isinstance(rounds, int):
            raise TypeError('kdf_rounds must be an integer')
        if rounds < 1:
            raise ValueError('kdf_rounds must be a positive integer')
        return KeySerializationEncryptionBuilder(self._format, _kdf_rounds=rounds, _hmac_hash=self._hmac_hash, _key_cert_algorithm=self._key_cert_algorithm)

    def hmac_hash(self, algorithm: HashAlgorithm) -> KeySerializationEncryptionBuilder:
        if self._format is not PrivateFormat.PKCS12:
            raise TypeError('hmac_hash only supported with PrivateFormat.PKCS12')
        if self._hmac_hash is not None:
            raise ValueError('hmac_hash already set')
        return KeySerializationEncryptionBuilder(self._format, _kdf_rounds=self._kdf_rounds, _hmac_hash=algorithm, _key_cert_algorithm=self._key_cert_algorithm)

    def key_cert_algorithm(self, algorithm: PBES) -> KeySerializationEncryptionBuilder:
        if self._format is not PrivateFormat.PKCS12:
            raise TypeError('key_cert_algorithm only supported with PrivateFormat.PKCS12')
        if self._key_cert_algorithm is not None:
            raise ValueError('key_cert_algorithm already set')
        return KeySerializationEncryptionBuilder(self._format, _kdf_rounds=self._kdf_rounds, _hmac_hash=self._hmac_hash, _key_cert_algorithm=algorithm)

    def build(self, password: bytes) -> KeySerializationEncryption:
        if not isinstance(password, bytes) or len(password) == 0:
            raise ValueError('Password must be 1 or more bytes.')
        return _KeySerializationEncryption(self._format, password, kdf_rounds=self._kdf_rounds, hmac_hash=self._hmac_hash, key_cert_algorithm=self._key_cert_algorithm)