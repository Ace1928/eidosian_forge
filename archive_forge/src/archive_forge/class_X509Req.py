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
class X509Req:
    """
    An X.509 certificate signing requests.
    """

    def __init__(self) -> None:
        req = _lib.X509_REQ_new()
        self._req = _ffi.gc(req, _lib.X509_REQ_free)
        self.set_version(0)

    def to_cryptography(self) -> x509.CertificateSigningRequest:
        """
        Export as a ``cryptography`` certificate signing request.

        :rtype: ``cryptography.x509.CertificateSigningRequest``

        .. versionadded:: 17.1.0
        """
        from cryptography.x509 import load_der_x509_csr
        der = dump_certificate_request(FILETYPE_ASN1, self)
        return load_der_x509_csr(der)

    @classmethod
    def from_cryptography(cls, crypto_req: x509.CertificateSigningRequest) -> 'X509Req':
        """
        Construct based on a ``cryptography`` *crypto_req*.

        :param crypto_req: A ``cryptography`` X.509 certificate signing request
        :type crypto_req: ``cryptography.x509.CertificateSigningRequest``

        :rtype: X509Req

        .. versionadded:: 17.1.0
        """
        if not isinstance(crypto_req, x509.CertificateSigningRequest):
            raise TypeError('Must be a certificate signing request')
        from cryptography.hazmat.primitives.serialization import Encoding
        der = crypto_req.public_bytes(Encoding.DER)
        return load_certificate_request(FILETYPE_ASN1, der)

    def set_pubkey(self, pkey: PKey) -> None:
        """
        Set the public key of the certificate signing request.

        :param pkey: The public key to use.
        :type pkey: :py:class:`PKey`

        :return: ``None``
        """
        set_result = _lib.X509_REQ_set_pubkey(self._req, pkey._pkey)
        _openssl_assert(set_result == 1)

    def get_pubkey(self) -> PKey:
        """
        Get the public key of the certificate signing request.

        :return: The public key.
        :rtype: :py:class:`PKey`
        """
        pkey = PKey.__new__(PKey)
        pkey._pkey = _lib.X509_REQ_get_pubkey(self._req)
        _openssl_assert(pkey._pkey != _ffi.NULL)
        pkey._pkey = _ffi.gc(pkey._pkey, _lib.EVP_PKEY_free)
        pkey._only_public = True
        return pkey

    def set_version(self, version: int) -> None:
        """
        Set the version subfield (RFC 2986, section 4.1) of the certificate
        request.

        :param int version: The version number.
        :return: ``None``
        """
        if not isinstance(version, int):
            raise TypeError('version must be an int')
        if version != 0:
            raise ValueError('Invalid version. The only valid version for X509Req is 0.')
        set_result = _lib.X509_REQ_set_version(self._req, version)
        _openssl_assert(set_result == 1)

    def get_version(self) -> int:
        """
        Get the version subfield (RFC 2459, section 4.1.2.1) of the certificate
        request.

        :return: The value of the version subfield.
        :rtype: :py:class:`int`
        """
        return _lib.X509_REQ_get_version(self._req)

    def get_subject(self) -> X509Name:
        """
        Return the subject of this certificate signing request.

        This creates a new :class:`X509Name` that wraps the underlying subject
        name field on the certificate signing request. Modifying it will modify
        the underlying signing request, and will have the effect of modifying
        any other :class:`X509Name` that refers to this subject.

        :return: The subject of this certificate signing request.
        :rtype: :class:`X509Name`
        """
        name = X509Name.__new__(X509Name)
        name._name = _lib.X509_REQ_get_subject_name(self._req)
        _openssl_assert(name._name != _ffi.NULL)
        name._owner = self
        return name

    def add_extensions(self, extensions: Iterable[X509Extension]) -> None:
        """
        Add extensions to the certificate signing request.

        :param extensions: The X.509 extensions to add.
        :type extensions: iterable of :py:class:`X509Extension`
        :return: ``None``
        """
        stack = _lib.sk_X509_EXTENSION_new_null()
        _openssl_assert(stack != _ffi.NULL)
        stack = _ffi.gc(stack, _lib.sk_X509_EXTENSION_free)
        for ext in extensions:
            if not isinstance(ext, X509Extension):
                raise ValueError('One of the elements is not an X509Extension')
            _lib.sk_X509_EXTENSION_push(stack, ext._extension)
        add_result = _lib.X509_REQ_add_extensions(self._req, stack)
        _openssl_assert(add_result == 1)

    def get_extensions(self) -> List[X509Extension]:
        """
        Get X.509 extensions in the certificate signing request.

        :return: The X.509 extensions in this request.
        :rtype: :py:class:`list` of :py:class:`X509Extension` objects.

        .. versionadded:: 0.15
        """
        exts = []
        native_exts_obj = _lib.X509_REQ_get_extensions(self._req)
        native_exts_obj = _ffi.gc(native_exts_obj, lambda x: _lib.sk_X509_EXTENSION_pop_free(x, _ffi.addressof(_lib._original_lib, 'X509_EXTENSION_free')))
        for i in range(_lib.sk_X509_EXTENSION_num(native_exts_obj)):
            ext = X509Extension.__new__(X509Extension)
            extension = _lib.X509_EXTENSION_dup(_lib.sk_X509_EXTENSION_value(native_exts_obj, i))
            ext._extension = _ffi.gc(extension, _lib.X509_EXTENSION_free)
            exts.append(ext)
        return exts

    def sign(self, pkey: PKey, digest: str) -> None:
        """
        Sign the certificate signing request with this key and digest type.

        :param pkey: The key pair to sign with.
        :type pkey: :py:class:`PKey`
        :param digest: The name of the message digest to use for the signature,
            e.g. :py:data:`"sha256"`.
        :type digest: :py:class:`str`
        :return: ``None``
        """
        if pkey._only_public:
            raise ValueError('Key has only public part')
        if not pkey._initialized:
            raise ValueError('Key is uninitialized')
        digest_obj = _lib.EVP_get_digestbyname(_byte_string(digest))
        if digest_obj == _ffi.NULL:
            raise ValueError('No such digest method')
        sign_result = _lib.X509_REQ_sign(self._req, pkey._pkey, digest_obj)
        _openssl_assert(sign_result > 0)

    def verify(self, pkey: PKey) -> bool:
        """
        Verifies the signature on this certificate signing request.

        :param PKey key: A public key.

        :return: ``True`` if the signature is correct.
        :rtype: bool

        :raises OpenSSL.crypto.Error: If the signature is invalid or there is a
            problem verifying the signature.
        """
        if not isinstance(pkey, PKey):
            raise TypeError('pkey must be a PKey instance')
        result = _lib.X509_REQ_verify(self._req, pkey._pkey)
        if result <= 0:
            _raise_current_error()
        return result