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
def _evp_pkey_to_public_key(self, evp_pkey) -> PublicKeyTypes:
    """
        Return the appropriate type of PublicKey given an evp_pkey cdata
        pointer.
        """
    key_type = self._lib.EVP_PKEY_id(evp_pkey)
    if key_type == self._lib.EVP_PKEY_RSA:
        rsa_cdata = self._lib.EVP_PKEY_get1_RSA(evp_pkey)
        self.openssl_assert(rsa_cdata != self._ffi.NULL)
        rsa_cdata = self._ffi.gc(rsa_cdata, self._lib.RSA_free)
        return _RSAPublicKey(self, rsa_cdata, evp_pkey)
    elif key_type == self._lib.EVP_PKEY_RSA_PSS and (not self._lib.CRYPTOGRAPHY_IS_LIBRESSL) and (not self._lib.CRYPTOGRAPHY_IS_BORINGSSL) and (not self._lib.CRYPTOGRAPHY_OPENSSL_LESS_THAN_111E):
        rsa_cdata = self._lib.EVP_PKEY_get1_RSA(evp_pkey)
        self.openssl_assert(rsa_cdata != self._ffi.NULL)
        rsa_cdata = self._ffi.gc(rsa_cdata, self._lib.RSA_free)
        bio = self._create_mem_bio_gc()
        res = self._lib.i2d_RSAPublicKey_bio(bio, rsa_cdata)
        self.openssl_assert(res == 1)
        return self.load_der_public_key(self._read_mem_bio(bio))
    elif key_type == self._lib.EVP_PKEY_DSA:
        return rust_openssl.dsa.public_key_from_ptr(int(self._ffi.cast('uintptr_t', evp_pkey)))
    elif key_type == self._lib.EVP_PKEY_EC:
        ec_cdata = self._lib.EVP_PKEY_get1_EC_KEY(evp_pkey)
        if ec_cdata == self._ffi.NULL:
            errors = self._consume_errors()
            raise ValueError('Unable to load EC key', errors)
        ec_cdata = self._ffi.gc(ec_cdata, self._lib.EC_KEY_free)
        return _EllipticCurvePublicKey(self, ec_cdata, evp_pkey)
    elif key_type in self._dh_types:
        return rust_openssl.dh.public_key_from_ptr(int(self._ffi.cast('uintptr_t', evp_pkey)))
    elif key_type == getattr(self._lib, 'EVP_PKEY_ED25519', None):
        return rust_openssl.ed25519.public_key_from_ptr(int(self._ffi.cast('uintptr_t', evp_pkey)))
    elif key_type == getattr(self._lib, 'EVP_PKEY_X448', None):
        return rust_openssl.x448.public_key_from_ptr(int(self._ffi.cast('uintptr_t', evp_pkey)))
    elif key_type == self._lib.EVP_PKEY_X25519:
        return rust_openssl.x25519.public_key_from_ptr(int(self._ffi.cast('uintptr_t', evp_pkey)))
    elif key_type == getattr(self._lib, 'EVP_PKEY_ED448', None):
        return rust_openssl.ed448.public_key_from_ptr(int(self._ffi.cast('uintptr_t', evp_pkey)))
    else:
        raise UnsupportedAlgorithm('Unsupported key type.')