from __future__ import annotations
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
def _check_key_infinity(backend: Backend, ec_cdata) -> None:
    point = backend._lib.EC_KEY_get0_public_key(ec_cdata)
    backend.openssl_assert(point != backend._ffi.NULL)
    group = backend._lib.EC_KEY_get0_group(ec_cdata)
    backend.openssl_assert(group != backend._ffi.NULL)
    if backend._lib.EC_POINT_is_at_infinity(group, point):
        raise ValueError('Cannot load an EC public key where the point is at infinity')