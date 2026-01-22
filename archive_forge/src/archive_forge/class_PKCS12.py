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
class PKCS12:
    """
    A PKCS #12 archive.
    """

    def __init__(self) -> None:
        self._pkey: Optional[PKey] = None
        self._cert: Optional[X509] = None
        self._cacerts: Optional[List[X509]] = None
        self._friendlyname: Optional[bytes] = None

    def get_certificate(self) -> Optional[X509]:
        """
        Get the certificate in the PKCS #12 structure.

        :return: The certificate, or :py:const:`None` if there is none.
        :rtype: :py:class:`X509` or :py:const:`None`
        """
        return self._cert

    def set_certificate(self, cert: X509) -> None:
        """
        Set the certificate in the PKCS #12 structure.

        :param cert: The new certificate, or :py:const:`None` to unset it.
        :type cert: :py:class:`X509` or :py:const:`None`

        :return: ``None``
        """
        if not isinstance(cert, X509):
            raise TypeError('cert must be an X509 instance')
        self._cert = cert

    def get_privatekey(self) -> Optional[PKey]:
        """
        Get the private key in the PKCS #12 structure.

        :return: The private key, or :py:const:`None` if there is none.
        :rtype: :py:class:`PKey`
        """
        return self._pkey

    def set_privatekey(self, pkey: PKey) -> None:
        """
        Set the certificate portion of the PKCS #12 structure.

        :param pkey: The new private key, or :py:const:`None` to unset it.
        :type pkey: :py:class:`PKey` or :py:const:`None`

        :return: ``None``
        """
        if not isinstance(pkey, PKey):
            raise TypeError('pkey must be a PKey instance')
        self._pkey = pkey

    def get_ca_certificates(self) -> Optional[Tuple[X509, ...]]:
        """
        Get the CA certificates in the PKCS #12 structure.

        :return: A tuple with the CA certificates in the chain, or
            :py:const:`None` if there are none.
        :rtype: :py:class:`tuple` of :py:class:`X509` or :py:const:`None`
        """
        if self._cacerts is not None:
            return tuple(self._cacerts)
        return None

    def set_ca_certificates(self, cacerts: Optional[Iterable[X509]]) -> None:
        """
        Replace or set the CA certificates within the PKCS12 object.

        :param cacerts: The new CA certificates, or :py:const:`None` to unset
            them.
        :type cacerts: An iterable of :py:class:`X509` or :py:const:`None`

        :return: ``None``
        """
        if cacerts is None:
            self._cacerts = None
        else:
            cacerts = list(cacerts)
            for cert in cacerts:
                if not isinstance(cert, X509):
                    raise TypeError('iterable must only contain X509 instances')
            self._cacerts = cacerts

    def set_friendlyname(self, name: Optional[bytes]) -> None:
        """
        Set the friendly name in the PKCS #12 structure.

        :param name: The new friendly name, or :py:const:`None` to unset.
        :type name: :py:class:`bytes` or :py:const:`None`

        :return: ``None``
        """
        if name is None:
            self._friendlyname = None
        elif not isinstance(name, bytes):
            raise TypeError('name must be a byte string or None (not %r)' % (name,))
        self._friendlyname = name

    def get_friendlyname(self) -> Optional[bytes]:
        """
        Get the friendly name in the PKCS# 12 structure.

        :returns: The friendly name,  or :py:const:`None` if there is none.
        :rtype: :py:class:`bytes` or :py:const:`None`
        """
        return self._friendlyname

    def export(self, passphrase: Optional[bytes]=None, iter: int=2048, maciter: int=1) -> bytes:
        """
        Dump a PKCS12 object as a string.

        For more information, see the :c:func:`PKCS12_create` man page.

        :param passphrase: The passphrase used to encrypt the structure. Unlike
            some other passphrase arguments, this *must* be a string, not a
            callback.
        :type passphrase: :py:data:`bytes`

        :param iter: Number of times to repeat the encryption step.
        :type iter: :py:data:`int`

        :param maciter: Number of times to repeat the MAC step.
        :type maciter: :py:data:`int`

        :return: The string representation of the PKCS #12 structure.
        :rtype:
        """
        passphrase = _text_to_bytes_and_warn('passphrase', passphrase)
        if self._cacerts is None:
            cacerts = _ffi.NULL
        else:
            cacerts = _lib.sk_X509_new_null()
            cacerts = _ffi.gc(cacerts, _lib.sk_X509_free)
            for cert in self._cacerts:
                _lib.sk_X509_push(cacerts, cert._x509)
        if passphrase is None:
            passphrase = _ffi.NULL
        friendlyname = self._friendlyname
        if friendlyname is None:
            friendlyname = _ffi.NULL
        if self._pkey is None:
            pkey = _ffi.NULL
        else:
            pkey = self._pkey._pkey
        if self._cert is None:
            cert = _ffi.NULL
        else:
            cert = self._cert._x509
        pkcs12 = _lib.PKCS12_create(passphrase, friendlyname, pkey, cert, cacerts, _lib.NID_pbe_WithSHA1And3_Key_TripleDES_CBC, _lib.NID_pbe_WithSHA1And3_Key_TripleDES_CBC, iter, maciter, 0)
        if pkcs12 == _ffi.NULL:
            _raise_current_error()
        pkcs12 = _ffi.gc(pkcs12, _lib.PKCS12_free)
        bio = _new_mem_buf()
        _lib.i2d_PKCS12_bio(bio, pkcs12)
        return _bio_to_string(bio)