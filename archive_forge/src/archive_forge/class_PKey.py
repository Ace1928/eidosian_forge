import calendar
import datetime
import functools
from base64 import b16encode
from functools import partial
from os import PathLike
from typing import (
from cryptography import utils, x509
from cryptography.hazmat.primitives.asymmetric import (
from OpenSSL._util import (
class PKey:
    """
    A class representing an DSA or RSA public key or key pair.
    """
    _only_public = False
    _initialized = True

    def __init__(self) -> None:
        pkey = _lib.EVP_PKEY_new()
        self._pkey = _ffi.gc(pkey, _lib.EVP_PKEY_free)
        self._initialized = False

    def to_cryptography_key(self) -> _Key:
        """
        Export as a ``cryptography`` key.

        :rtype: One of ``cryptography``'s `key interfaces`_.

        .. _key interfaces: https://cryptography.io/en/latest/hazmat/            primitives/asymmetric/rsa/#key-interfaces

        .. versionadded:: 16.1.0
        """
        from cryptography.hazmat.primitives.serialization import load_der_private_key, load_der_public_key
        if self._only_public:
            der = dump_publickey(FILETYPE_ASN1, self)
            return load_der_public_key(der)
        else:
            der = dump_privatekey(FILETYPE_ASN1, self)
            return load_der_private_key(der, None)

    @classmethod
    def from_cryptography_key(cls, crypto_key: _Key) -> 'PKey':
        """
        Construct based on a ``cryptography`` *crypto_key*.

        :param crypto_key: A ``cryptography`` key.
        :type crypto_key: One of ``cryptography``'s `key interfaces`_.

        :rtype: PKey

        .. versionadded:: 16.1.0
        """
        if not isinstance(crypto_key, (rsa.RSAPublicKey, rsa.RSAPrivateKey, dsa.DSAPublicKey, dsa.DSAPrivateKey, ec.EllipticCurvePrivateKey, ed25519.Ed25519PrivateKey, ed448.Ed448PrivateKey)):
            raise TypeError('Unsupported key type')
        from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat
        if isinstance(crypto_key, (rsa.RSAPublicKey, dsa.DSAPublicKey)):
            return load_publickey(FILETYPE_ASN1, crypto_key.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo))
        else:
            der = crypto_key.private_bytes(Encoding.DER, PrivateFormat.PKCS8, NoEncryption())
            return load_privatekey(FILETYPE_ASN1, der)

    def generate_key(self, type: int, bits: int) -> None:
        """
        Generate a key pair of the given type, with the given number of bits.

        This generates a key "into" the this object.

        :param type: The key type.
        :type type: :py:data:`TYPE_RSA` or :py:data:`TYPE_DSA`
        :param bits: The number of bits.
        :type bits: :py:data:`int` ``>= 0``
        :raises TypeError: If :py:data:`type` or :py:data:`bits` isn't
            of the appropriate type.
        :raises ValueError: If the number of bits isn't an integer of
            the appropriate size.
        :return: ``None``
        """
        if not isinstance(type, int):
            raise TypeError('type must be an integer')
        if not isinstance(bits, int):
            raise TypeError('bits must be an integer')
        if type == TYPE_RSA:
            if bits <= 0:
                raise ValueError('Invalid number of bits')
            exponent = _lib.BN_new()
            exponent = _ffi.gc(exponent, _lib.BN_free)
            _lib.BN_set_word(exponent, _lib.RSA_F4)
            rsa = _lib.RSA_new()
            result = _lib.RSA_generate_key_ex(rsa, bits, exponent, _ffi.NULL)
            _openssl_assert(result == 1)
            result = _lib.EVP_PKEY_assign_RSA(self._pkey, rsa)
            _openssl_assert(result == 1)
        elif type == TYPE_DSA:
            dsa = _lib.DSA_new()
            _openssl_assert(dsa != _ffi.NULL)
            dsa = _ffi.gc(dsa, _lib.DSA_free)
            res = _lib.DSA_generate_parameters_ex(dsa, bits, _ffi.NULL, 0, _ffi.NULL, _ffi.NULL, _ffi.NULL)
            _openssl_assert(res == 1)
            _openssl_assert(_lib.DSA_generate_key(dsa) == 1)
            _openssl_assert(_lib.EVP_PKEY_set1_DSA(self._pkey, dsa) == 1)
        else:
            raise Error('No such key type')
        self._initialized = True

    def check(self) -> bool:
        """
        Check the consistency of an RSA private key.

        This is the Python equivalent of OpenSSL's ``RSA_check_key``.

        :return: ``True`` if key is consistent.

        :raise OpenSSL.crypto.Error: if the key is inconsistent.

        :raise TypeError: if the key is of a type which cannot be checked.
            Only RSA keys can currently be checked.
        """
        if self._only_public:
            raise TypeError('public key only')
        if _lib.EVP_PKEY_type(self.type()) != _lib.EVP_PKEY_RSA:
            raise TypeError('Only RSA keys can currently be checked.')
        rsa = _lib.EVP_PKEY_get1_RSA(self._pkey)
        rsa = _ffi.gc(rsa, _lib.RSA_free)
        result = _lib.RSA_check_key(rsa)
        if result == 1:
            return True
        _raise_current_error()

    def type(self) -> int:
        """
        Returns the type of the key

        :return: The type of the key.
        """
        return _lib.EVP_PKEY_id(self._pkey)

    def bits(self) -> int:
        """
        Returns the number of bits of the key

        :return: The number of bits of the key.
        """
        return _lib.EVP_PKEY_bits(self._pkey)