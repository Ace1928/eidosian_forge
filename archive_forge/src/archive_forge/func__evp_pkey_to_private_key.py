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
def _evp_pkey_to_private_key(self, evp_pkey, unsafe_skip_rsa_key_validation: bool) -> PrivateKeyTypes:
    """
        Return the appropriate type of PrivateKey given an evp_pkey cdata
        pointer.
        """
    key_type = self._lib.EVP_PKEY_id(evp_pkey)
    if key_type == self._lib.EVP_PKEY_RSA:
        rsa_cdata = self._lib.EVP_PKEY_get1_RSA(evp_pkey)
        self.openssl_assert(rsa_cdata != self._ffi.NULL)
        rsa_cdata = self._ffi.gc(rsa_cdata, self._lib.RSA_free)
        return _RSAPrivateKey(self, rsa_cdata, evp_pkey, unsafe_skip_rsa_key_validation=unsafe_skip_rsa_key_validation)
    elif key_type == self._lib.EVP_PKEY_RSA_PSS and (not self._lib.CRYPTOGRAPHY_IS_LIBRESSL) and (not self._lib.CRYPTOGRAPHY_IS_BORINGSSL) and (not self._lib.CRYPTOGRAPHY_OPENSSL_LESS_THAN_111E):
        rsa_cdata = self._lib.EVP_PKEY_get1_RSA(evp_pkey)
        self.openssl_assert(rsa_cdata != self._ffi.NULL)
        rsa_cdata = self._ffi.gc(rsa_cdata, self._lib.RSA_free)
        bio = self._create_mem_bio_gc()
        res = self._lib.i2d_RSAPrivateKey_bio(bio, rsa_cdata)
        self.openssl_assert(res == 1)
        return self.load_der_private_key(self._read_mem_bio(bio), password=None, unsafe_skip_rsa_key_validation=unsafe_skip_rsa_key_validation)
    elif key_type == self._lib.EVP_PKEY_DSA:
        return rust_openssl.dsa.private_key_from_ptr(int(self._ffi.cast('uintptr_t', evp_pkey)))
    elif key_type == self._lib.EVP_PKEY_EC:
        ec_cdata = self._lib.EVP_PKEY_get1_EC_KEY(evp_pkey)
        self.openssl_assert(ec_cdata != self._ffi.NULL)
        ec_cdata = self._ffi.gc(ec_cdata, self._lib.EC_KEY_free)
        return _EllipticCurvePrivateKey(self, ec_cdata, evp_pkey)
    elif key_type in self._dh_types:
        return rust_openssl.dh.private_key_from_ptr(int(self._ffi.cast('uintptr_t', evp_pkey)))
    elif key_type == getattr(self._lib, 'EVP_PKEY_ED25519', None):
        return rust_openssl.ed25519.private_key_from_ptr(int(self._ffi.cast('uintptr_t', evp_pkey)))
    elif key_type == getattr(self._lib, 'EVP_PKEY_X448', None):
        return rust_openssl.x448.private_key_from_ptr(int(self._ffi.cast('uintptr_t', evp_pkey)))
    elif key_type == self._lib.EVP_PKEY_X25519:
        return rust_openssl.x25519.private_key_from_ptr(int(self._ffi.cast('uintptr_t', evp_pkey)))
    elif key_type == getattr(self._lib, 'EVP_PKEY_ED448', None):
        return rust_openssl.ed448.private_key_from_ptr(int(self._ffi.cast('uintptr_t', evp_pkey)))
    else:
        raise UnsupportedAlgorithm('Unsupported key type.')