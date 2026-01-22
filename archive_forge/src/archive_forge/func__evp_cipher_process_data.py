from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag
def _evp_cipher_process_data(backend: Backend, ctx, data: bytes) -> bytes:
    outlen = backend._ffi.new('int *')
    buf = backend._ffi.new('unsigned char[]', len(data))
    data_ptr = backend._ffi.from_buffer(data)
    res = backend._lib.EVP_CipherUpdate(ctx, buf, outlen, data_ptr, len(data))
    if res == 0:
        backend._consume_errors()
        raise InvalidTag
    return backend._ffi.buffer(buf, outlen[0])[:]