from __future__ import annotations
import os
import typing
from cryptography import exceptions, utils
from cryptography.hazmat.backends.openssl import aead
from cryptography.hazmat.backends.openssl.backend import backend
from cryptography.hazmat.bindings._rust import FixedPool
class AESSIV:
    _MAX_SIZE = 2 ** 31 - 1

    def __init__(self, key: bytes):
        utils._check_byteslike('key', key)
        if len(key) not in (32, 48, 64):
            raise ValueError('AESSIV key must be 256, 384, or 512 bits.')
        self._key = key
        if not backend.aead_cipher_supported(self):
            raise exceptions.UnsupportedAlgorithm('AES-SIV is not supported by this version of OpenSSL', exceptions._Reasons.UNSUPPORTED_CIPHER)

    @classmethod
    def generate_key(cls, bit_length: int) -> bytes:
        if not isinstance(bit_length, int):
            raise TypeError('bit_length must be an integer')
        if bit_length not in (256, 384, 512):
            raise ValueError('bit_length must be 256, 384, or 512')
        return os.urandom(bit_length // 8)

    def encrypt(self, data: bytes, associated_data: typing.Optional[typing.List[bytes]]) -> bytes:
        if associated_data is None:
            associated_data = []
        self._check_params(data, associated_data)
        if len(data) > self._MAX_SIZE or any((len(ad) > self._MAX_SIZE for ad in associated_data)):
            raise OverflowError('Data or associated data too long. Max 2**31 - 1 bytes')
        return aead._encrypt(backend, self, b'', data, associated_data, 16)

    def decrypt(self, data: bytes, associated_data: typing.Optional[typing.List[bytes]]) -> bytes:
        if associated_data is None:
            associated_data = []
        self._check_params(data, associated_data)
        return aead._decrypt(backend, self, b'', data, associated_data, 16)

    def _check_params(self, data: bytes, associated_data: typing.List[bytes]) -> None:
        utils._check_byteslike('data', data)
        if len(data) == 0:
            raise ValueError('data must not be zero length')
        if not isinstance(associated_data, list):
            raise TypeError('associated_data must be a list of bytes-like objects or None')
        for x in associated_data:
            utils._check_byteslike('associated_data elements', x)