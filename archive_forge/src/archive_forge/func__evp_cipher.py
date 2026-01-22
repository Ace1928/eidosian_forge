from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag
def _evp_cipher(cipher_name: bytes, backend: Backend):
    if cipher_name.endswith(b'-siv'):
        evp_cipher = backend._lib.EVP_CIPHER_fetch(backend._ffi.NULL, cipher_name, backend._ffi.NULL)
        backend.openssl_assert(evp_cipher != backend._ffi.NULL)
        evp_cipher = backend._ffi.gc(evp_cipher, backend._lib.EVP_CIPHER_free)
    else:
        evp_cipher = backend._lib.EVP_get_cipherbyname(cipher_name)
        backend.openssl_assert(evp_cipher != backend._ffi.NULL)
    return evp_cipher