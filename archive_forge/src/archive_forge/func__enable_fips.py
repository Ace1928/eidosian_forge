from __future__ import annotations
import os
import sys
import threading
import types
import typing
import warnings
import cryptography
from cryptography.exceptions import InternalError
from cryptography.hazmat.bindings._rust import _openssl, openssl
from cryptography.hazmat.bindings.openssl._conditional import CONDITIONAL_NAMES
def _enable_fips(self) -> None:
    _openssl_assert(self.lib, self.lib.CRYPTOGRAPHY_OPENSSL_300_OR_GREATER)
    self._base_provider = self.lib.OSSL_PROVIDER_load(self.ffi.NULL, b'base')
    _openssl_assert(self.lib, self._base_provider != self.ffi.NULL)
    self.lib._fips_provider = self.lib.OSSL_PROVIDER_load(self.ffi.NULL, b'fips')
    _openssl_assert(self.lib, self.lib._fips_provider != self.ffi.NULL)
    res = self.lib.EVP_default_properties_enable_fips(self.ffi.NULL, 1)
    _openssl_assert(self.lib, res == 1)