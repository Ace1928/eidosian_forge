from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag
def _evp_aead_encrypt(backend: Backend, cipher: _AEADTypes, nonce: bytes, data: bytes, associated_data: typing.List[bytes], tag_length: int, ctx: typing.Any) -> bytes:
    assert ctx is not None
    aead_cipher = _evp_aead_get_cipher(backend, cipher)
    assert aead_cipher is not None
    out_len = backend._ffi.new('size_t *')
    max_out_len = len(data) + backend._lib.EVP_AEAD_max_overhead(aead_cipher)
    out_buf = backend._ffi.new('uint8_t[]', max_out_len)
    data_ptr = backend._ffi.from_buffer(data)
    nonce_ptr = backend._ffi.from_buffer(nonce)
    aad = b''.join(associated_data)
    aad_ptr = backend._ffi.from_buffer(aad)
    res = backend._lib.EVP_AEAD_CTX_seal(ctx, out_buf, out_len, max_out_len, nonce_ptr, len(nonce), data_ptr, len(data), aad_ptr, len(aad))
    backend.openssl_assert(res == 1)
    encrypted_data = backend._ffi.buffer(out_buf, out_len[0])[:]
    return encrypted_data