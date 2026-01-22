from __future__ import annotations
import threading
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.asymmetric.padding import (
from cryptography.hazmat.primitives.asymmetric.rsa import (
def _rsa_sig_recover(backend: Backend, padding: AsymmetricPadding, algorithm: typing.Optional[hashes.HashAlgorithm], public_key: _RSAPublicKey, signature: bytes) -> bytes:
    pkey_ctx = _rsa_sig_setup(backend, padding, algorithm, public_key, backend._lib.EVP_PKEY_verify_recover_init)
    maxlen = backend._lib.EVP_PKEY_size(public_key._evp_pkey)
    backend.openssl_assert(maxlen > 0)
    buf = backend._ffi.new('unsigned char[]', maxlen)
    buflen = backend._ffi.new('size_t *', maxlen)
    res = backend._lib.EVP_PKEY_verify_recover(pkey_ctx, buf, buflen, signature, len(signature))
    resbuf = backend._ffi.buffer(buf)[:buflen[0]]
    backend._lib.ERR_clear_error()
    if res != 1:
        raise InvalidSignature
    return resbuf