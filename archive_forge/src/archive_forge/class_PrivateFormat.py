from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.hazmat.primitives.hashes import HashAlgorithm
class PrivateFormat(utils.Enum):
    PKCS8 = 'PKCS8'
    TraditionalOpenSSL = 'TraditionalOpenSSL'
    Raw = 'Raw'
    OpenSSH = 'OpenSSH'
    PKCS12 = 'PKCS12'

    def encryption_builder(self) -> KeySerializationEncryptionBuilder:
        if self not in (PrivateFormat.OpenSSH, PrivateFormat.PKCS12):
            raise ValueError('encryption_builder only supported with PrivateFormat.OpenSSH and PrivateFormat.PKCS12')
        return KeySerializationEncryptionBuilder(self)