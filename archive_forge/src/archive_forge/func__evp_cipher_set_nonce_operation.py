from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag
def _evp_cipher_set_nonce_operation(backend, ctx, nonce: bytes, operation: int) -> None:
    nonce_ptr = backend._ffi.from_buffer(nonce)
    res = backend._lib.EVP_CipherInit_ex(ctx, backend._ffi.NULL, backend._ffi.NULL, backend._ffi.NULL, nonce_ptr, int(operation == _ENCRYPT))
    backend.openssl_assert(res != 0)