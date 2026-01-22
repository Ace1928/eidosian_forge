from __future__ import annotations
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
def _encode_point(self, format: serialization.PublicFormat) -> bytes:
    if format is serialization.PublicFormat.CompressedPoint:
        conversion = self._backend._lib.POINT_CONVERSION_COMPRESSED
    else:
        assert format is serialization.PublicFormat.UncompressedPoint
        conversion = self._backend._lib.POINT_CONVERSION_UNCOMPRESSED
    group = self._backend._lib.EC_KEY_get0_group(self._ec_key)
    self._backend.openssl_assert(group != self._backend._ffi.NULL)
    point = self._backend._lib.EC_KEY_get0_public_key(self._ec_key)
    self._backend.openssl_assert(point != self._backend._ffi.NULL)
    with self._backend._tmp_bn_ctx() as bn_ctx:
        buflen = self._backend._lib.EC_POINT_point2oct(group, point, conversion, self._backend._ffi.NULL, 0, bn_ctx)
        self._backend.openssl_assert(buflen > 0)
        buf = self._backend._ffi.new('char[]', buflen)
        res = self._backend._lib.EC_POINT_point2oct(group, point, conversion, buf, buflen, bn_ctx)
        self._backend.openssl_assert(buflen == res)
    return self._backend._ffi.buffer(buf)[:]