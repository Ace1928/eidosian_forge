from __future__ import annotations
import threading
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.asymmetric.padding import (
from cryptography.hazmat.primitives.asymmetric.rsa import (
class _RSAPublicKey(RSAPublicKey):
    _evp_pkey: object
    _rsa_cdata: object
    _key_size: int

    def __init__(self, backend: Backend, rsa_cdata, evp_pkey):
        self._backend = backend
        self._rsa_cdata = rsa_cdata
        self._evp_pkey = evp_pkey
        n = self._backend._ffi.new('BIGNUM **')
        self._backend._lib.RSA_get0_key(self._rsa_cdata, n, self._backend._ffi.NULL, self._backend._ffi.NULL)
        self._backend.openssl_assert(n[0] != self._backend._ffi.NULL)
        self._key_size = self._backend._lib.BN_num_bits(n[0])

    @property
    def key_size(self) -> int:
        return self._key_size

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _RSAPublicKey):
            return NotImplemented
        return self._backend._lib.EVP_PKEY_cmp(self._evp_pkey, other._evp_pkey) == 1

    def encrypt(self, plaintext: bytes, padding: AsymmetricPadding) -> bytes:
        return _enc_dec_rsa(self._backend, self, plaintext, padding)

    def public_numbers(self) -> RSAPublicNumbers:
        n = self._backend._ffi.new('BIGNUM **')
        e = self._backend._ffi.new('BIGNUM **')
        self._backend._lib.RSA_get0_key(self._rsa_cdata, n, e, self._backend._ffi.NULL)
        self._backend.openssl_assert(n[0] != self._backend._ffi.NULL)
        self._backend.openssl_assert(e[0] != self._backend._ffi.NULL)
        return RSAPublicNumbers(e=self._backend._bn_to_int(e[0]), n=self._backend._bn_to_int(n[0]))

    def public_bytes(self, encoding: serialization.Encoding, format: serialization.PublicFormat) -> bytes:
        return self._backend._public_key_bytes(encoding, format, self, self._evp_pkey, self._rsa_cdata)

    def verify(self, signature: bytes, data: bytes, padding: AsymmetricPadding, algorithm: typing.Union[asym_utils.Prehashed, hashes.HashAlgorithm]) -> None:
        data, algorithm = _calculate_digest_and_algorithm(data, algorithm)
        _rsa_sig_verify(self._backend, padding, algorithm, self, signature, data)

    def recover_data_from_signature(self, signature: bytes, padding: AsymmetricPadding, algorithm: typing.Optional[hashes.HashAlgorithm]) -> bytes:
        if isinstance(algorithm, asym_utils.Prehashed):
            raise TypeError('Prehashed is only supported in the sign and verify methods. It cannot be used with recover_data_from_signature.')
        return _rsa_sig_recover(self._backend, padding, algorithm, self, signature)