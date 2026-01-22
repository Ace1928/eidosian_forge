from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag
def _encrypt(backend: Backend, cipher: _AEADTypes, nonce: bytes, data: bytes, associated_data: typing.List[bytes], tag_length: int, ctx: typing.Any=None) -> bytes:
    if _is_evp_aead_supported_cipher(backend, cipher):
        return _evp_aead_encrypt(backend, cipher, nonce, data, associated_data, tag_length, ctx)
    else:
        return _evp_cipher_encrypt(backend, cipher, nonce, data, associated_data, tag_length, ctx)