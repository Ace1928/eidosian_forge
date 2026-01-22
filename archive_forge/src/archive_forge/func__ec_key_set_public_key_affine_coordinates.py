from __future__ import annotations
import collections
import contextlib
import itertools
import typing
from contextlib import contextmanager
from cryptography import utils, x509
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.backends.openssl import aead
from cryptography.hazmat.backends.openssl.ciphers import _CipherContext
from cryptography.hazmat.backends.openssl.cmac import _CMACContext
from cryptography.hazmat.backends.openssl.ec import (
from cryptography.hazmat.backends.openssl.rsa import (
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.bindings.openssl import binding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives._asymmetric import AsymmetricPadding
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.padding import (
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.hazmat.primitives.ciphers import (
from cryptography.hazmat.primitives.ciphers.algorithms import (
from cryptography.hazmat.primitives.ciphers.modes import (
from cryptography.hazmat.primitives.serialization import ssh
from cryptography.hazmat.primitives.serialization.pkcs12 import (
def _ec_key_set_public_key_affine_coordinates(self, ec_cdata, x: int, y: int, bn_ctx) -> None:
    """
        Sets the public key point in the EC_KEY context to the affine x and y
        values.
        """
    if x < 0 or y < 0:
        raise ValueError('Invalid EC key. Both x and y must be non-negative.')
    x = self._ffi.gc(self._int_to_bn(x), self._lib.BN_free)
    y = self._ffi.gc(self._int_to_bn(y), self._lib.BN_free)
    group = self._lib.EC_KEY_get0_group(ec_cdata)
    self.openssl_assert(group != self._ffi.NULL)
    point = self._lib.EC_POINT_new(group)
    self.openssl_assert(point != self._ffi.NULL)
    point = self._ffi.gc(point, self._lib.EC_POINT_free)
    res = self._lib.EC_POINT_set_affine_coordinates(group, point, x, y, bn_ctx)
    if res != 1:
        self._consume_errors()
        raise ValueError('Invalid EC key.')
    res = self._lib.EC_KEY_set_public_key(ec_cdata, point)
    self.openssl_assert(res == 1)