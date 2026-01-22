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
class X509StoreContext:
    """
    An X.509 store context.

    An X.509 store context is used to carry out the actual verification process
    of a certificate in a described context. For describing such a context, see
    :class:`X509Store`.

    :ivar _store_ctx: The underlying X509_STORE_CTX structure used by this
        instance.  It is dynamically allocated and automatically garbage
        collected.
    :ivar _store: See the ``store`` ``__init__`` parameter.
    :ivar _cert: See the ``certificate`` ``__init__`` parameter.
    :ivar _chain: See the ``chain`` ``__init__`` parameter.
    :param X509Store store: The certificates which will be trusted for the
        purposes of any verifications.
    :param X509 certificate: The certificate to be verified.
    :param chain: List of untrusted certificates that may be used for building
        the certificate chain. May be ``None``.
    :type chain: :class:`list` of :class:`X509`
    """

    def __init__(self, store: X509Store, certificate: X509, chain: Optional[Sequence[X509]]=None) -> None:
        store_ctx = _lib.X509_STORE_CTX_new()
        self._store_ctx = _ffi.gc(store_ctx, _lib.X509_STORE_CTX_free)
        self._store = store
        self._cert = certificate
        self._chain = self._build_certificate_stack(chain)
        self._init()

    @staticmethod
    def _build_certificate_stack(certificates: Optional[Sequence[X509]]) -> None:

        def cleanup(s: Any) -> None:
            for i in range(_lib.sk_X509_num(s)):
                x = _lib.sk_X509_value(s, i)
                _lib.X509_free(x)
            _lib.sk_X509_free(s)
        if certificates is None or len(certificates) == 0:
            return _ffi.NULL
        stack = _lib.sk_X509_new_null()
        _openssl_assert(stack != _ffi.NULL)
        stack = _ffi.gc(stack, cleanup)
        for cert in certificates:
            if not isinstance(cert, X509):
                raise TypeError('One of the elements is not an X509 instance')
            _openssl_assert(_lib.X509_up_ref(cert._x509) > 0)
            if _lib.sk_X509_push(stack, cert._x509) <= 0:
                _lib.X509_free(cert._x509)
                _raise_current_error()
        return stack

    def _init(self) -> None:
        """
        Set up the store context for a subsequent verification operation.

        Calling this method more than once without first calling
        :meth:`_cleanup` will leak memory.
        """
        ret = _lib.X509_STORE_CTX_init(self._store_ctx, self._store._store, self._cert._x509, self._chain)
        if ret <= 0:
            _raise_current_error()

    def _cleanup(self) -> None:
        """
        Internally cleans up the store context.

        The store context can then be reused with a new call to :meth:`_init`.
        """
        _lib.X509_STORE_CTX_cleanup(self._store_ctx)

    def _exception_from_context(self) -> X509StoreContextError:
        """
        Convert an OpenSSL native context error failure into a Python
        exception.

        When a call to native OpenSSL X509_verify_cert fails, additional
        information about the failure can be obtained from the store context.
        """
        message = _ffi.string(_lib.X509_verify_cert_error_string(_lib.X509_STORE_CTX_get_error(self._store_ctx))).decode('utf-8')
        errors = [_lib.X509_STORE_CTX_get_error(self._store_ctx), _lib.X509_STORE_CTX_get_error_depth(self._store_ctx), message]
        _x509 = _lib.X509_STORE_CTX_get_current_cert(self._store_ctx)
        _cert = _lib.X509_dup(_x509)
        pycert = X509._from_raw_x509_ptr(_cert)
        return X509StoreContextError(message, errors, pycert)

    def set_store(self, store: X509Store) -> None:
        """
        Set the context's X.509 store.

        .. versionadded:: 0.15

        :param X509Store store: The store description which will be used for
            the purposes of any *future* verifications.
        """
        self._store = store

    def verify_certificate(self) -> None:
        """
        Verify a certificate in a context.

        .. versionadded:: 0.15

        :raises X509StoreContextError: If an error occurred when validating a
          certificate in the context. Sets ``certificate`` attribute to
          indicate which certificate caused the error.
        """
        self._cleanup()
        self._init()
        ret = _lib.X509_verify_cert(self._store_ctx)
        self._cleanup()
        if ret <= 0:
            raise self._exception_from_context()

    def get_verified_chain(self) -> List[X509]:
        """
        Verify a certificate in a context and return the complete validated
        chain.

        :raises X509StoreContextError: If an error occurred when validating a
          certificate in the context. Sets ``certificate`` attribute to
          indicate which certificate caused the error.

        .. versionadded:: 20.0
        """
        self._cleanup()
        self._init()
        ret = _lib.X509_verify_cert(self._store_ctx)
        if ret <= 0:
            self._cleanup()
            raise self._exception_from_context()
        cert_stack = _lib.X509_STORE_CTX_get1_chain(self._store_ctx)
        _openssl_assert(cert_stack != _ffi.NULL)
        result = []
        for i in range(_lib.sk_X509_num(cert_stack)):
            cert = _lib.sk_X509_value(cert_stack, i)
            _openssl_assert(cert != _ffi.NULL)
            pycert = X509._from_raw_x509_ptr(cert)
            result.append(pycert)
        _lib.sk_X509_free(cert_stack)
        self._cleanup()
        return result