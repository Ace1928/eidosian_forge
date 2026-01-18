from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag, UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.primitives import ciphers
from cryptography.hazmat.primitives.ciphers import algorithms, modes
def update_into(self, data: bytes, buf: bytes) -> int:
    total_data_len = len(data)
    if len(buf) < total_data_len + self._block_size_bytes - 1:
        raise ValueError('buffer must be at least {} bytes for this payload'.format(len(data) + self._block_size_bytes - 1))
    data_processed = 0
    total_out = 0
    outlen = self._backend._ffi.new('int *')
    baseoutbuf = self._backend._ffi.from_buffer(buf, require_writable=True)
    baseinbuf = self._backend._ffi.from_buffer(data)
    while data_processed != total_data_len:
        outbuf = baseoutbuf + total_out
        inbuf = baseinbuf + data_processed
        inlen = min(self._MAX_CHUNK_SIZE, total_data_len - data_processed)
        res = self._backend._lib.EVP_CipherUpdate(self._ctx, outbuf, outlen, inbuf, inlen)
        if res == 0 and isinstance(self._mode, modes.XTS):
            self._backend._consume_errors()
            raise ValueError('In XTS mode you must supply at least a full block in the first update call. For AES this is 16 bytes.')
        else:
            self._backend.openssl_assert(res != 0)
        data_processed += inlen
        total_out += outlen[0]
    return total_out