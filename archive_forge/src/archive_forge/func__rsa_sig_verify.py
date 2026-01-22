from __future__ import annotations
import threading
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.asymmetric.padding import (
from cryptography.hazmat.primitives.asymmetric.rsa import (
def _rsa_sig_verify(backend: Backend, padding: AsymmetricPadding, algorithm: hashes.HashAlgorithm, public_key: _RSAPublicKey, signature: bytes, data: bytes) -> None:
    pkey_ctx = _rsa_sig_setup(backend, padding, algorithm, public_key, backend._lib.EVP_PKEY_verify_init)
    res = backend._lib.EVP_PKEY_verify(pkey_ctx, signature, len(signature), data, len(data))
    backend.openssl_assert(res >= 0)
    if res == 0:
        backend._consume_errors()
        raise InvalidSignature