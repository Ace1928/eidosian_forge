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
class X509Extension:
    """
    An X.509 v3 certificate extension.
    """

    def __init__(self, type_name: bytes, critical: bool, value: bytes, subject: Optional['X509']=None, issuer: Optional['X509']=None) -> None:
        """
        Initializes an X509 extension.

        :param type_name: The name of the type of extension_ to create.
        :type type_name: :py:data:`bytes`

        :param bool critical: A flag indicating whether this is a critical
            extension.

        :param value: The OpenSSL textual representation of the extension's
            value.
        :type value: :py:data:`bytes`

        :param subject: Optional X509 certificate to use as subject.
        :type subject: :py:class:`X509`

        :param issuer: Optional X509 certificate to use as issuer.
        :type issuer: :py:class:`X509`

        .. _extension: https://www.openssl.org/docs/manmaster/man5/
            x509v3_config.html#STANDARD-EXTENSIONS
        """
        ctx = _ffi.new('X509V3_CTX*')
        _lib.X509V3_set_ctx(ctx, _ffi.NULL, _ffi.NULL, _ffi.NULL, _ffi.NULL, 0)
        _lib.X509V3_set_ctx_nodb(ctx)
        if issuer is not None:
            if not isinstance(issuer, X509):
                raise TypeError('issuer must be an X509 instance')
            ctx.issuer_cert = issuer._x509
        if subject is not None:
            if not isinstance(subject, X509):
                raise TypeError('subject must be an X509 instance')
            ctx.subject_cert = subject._x509
        if critical:
            value = b'critical,' + value
        extension = _lib.X509V3_EXT_nconf(_ffi.NULL, ctx, type_name, value)
        if extension == _ffi.NULL:
            _raise_current_error()
        self._extension = _ffi.gc(extension, _lib.X509_EXTENSION_free)

    @property
    def _nid(self) -> Any:
        return _lib.OBJ_obj2nid(_lib.X509_EXTENSION_get_object(self._extension))
    _prefixes = {_lib.GEN_EMAIL: 'email', _lib.GEN_DNS: 'DNS', _lib.GEN_URI: 'URI'}

    def _subjectAltNameString(self) -> str:
        names = _ffi.cast('GENERAL_NAMES*', _lib.X509V3_EXT_d2i(self._extension))
        names = _ffi.gc(names, _lib.GENERAL_NAMES_free)
        parts = []
        for i in range(_lib.sk_GENERAL_NAME_num(names)):
            name = _lib.sk_GENERAL_NAME_value(names, i)
            try:
                label = self._prefixes[name.type]
            except KeyError:
                bio = _new_mem_buf()
                _lib.GENERAL_NAME_print(bio, name)
                parts.append(_bio_to_string(bio).decode('utf-8'))
            else:
                value = _ffi.buffer(name.d.ia5.data, name.d.ia5.length)[:].decode('utf-8')
                parts.append(label + ':' + value)
        return ', '.join(parts)

    def __str__(self) -> str:
        """
        :return: a nice text representation of the extension
        """
        if _lib.NID_subject_alt_name == self._nid:
            return self._subjectAltNameString()
        bio = _new_mem_buf()
        print_result = _lib.X509V3_EXT_print(bio, self._extension, 0, 0)
        _openssl_assert(print_result != 0)
        return _bio_to_string(bio).decode('utf-8')

    def get_critical(self) -> bool:
        """
        Returns the critical field of this X.509 extension.

        :return: The critical field.
        """
        return _lib.X509_EXTENSION_get_critical(self._extension)

    def get_short_name(self) -> bytes:
        """
        Returns the short type name of this X.509 extension.

        The result is a byte string such as :py:const:`b"basicConstraints"`.

        :return: The short type name.
        :rtype: :py:data:`bytes`

        .. versionadded:: 0.12
        """
        obj = _lib.X509_EXTENSION_get_object(self._extension)
        nid = _lib.OBJ_obj2nid(obj)
        buf = _lib.OBJ_nid2sn(nid)
        if buf != _ffi.NULL:
            return _ffi.string(buf)
        else:
            return b'UNDEF'

    def get_data(self) -> bytes:
        """
        Returns the data of the X509 extension, encoded as ASN.1.

        :return: The ASN.1 encoded data of this X509 extension.
        :rtype: :py:data:`bytes`

        .. versionadded:: 0.12
        """
        octet_result = _lib.X509_EXTENSION_get_data(self._extension)
        string_result = _ffi.cast('ASN1_STRING*', octet_result)
        char_result = _lib.ASN1_STRING_get0_data(string_result)
        result_length = _lib.ASN1_STRING_length(string_result)
        return _ffi.buffer(char_result, result_length)[:]