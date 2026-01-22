from __future__ import annotations
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
def _ecdsa_sig_sign(backend: Backend, private_key: _EllipticCurvePrivateKey, data: bytes) -> bytes:
    max_size = backend._lib.ECDSA_size(private_key._ec_key)
    backend.openssl_assert(max_size > 0)
    sigbuf = backend._ffi.new('unsigned char[]', max_size)
    siglen_ptr = backend._ffi.new('unsigned int[]', 1)
    res = backend._lib.ECDSA_sign(0, data, len(data), sigbuf, siglen_ptr, private_key._ec_key)
    backend.openssl_assert(res == 1)
    return backend._ffi.buffer(sigbuf)[:siglen_ptr[0]]