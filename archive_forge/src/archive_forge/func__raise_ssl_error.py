import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def _raise_ssl_error(self, ssl, result):
    if self._context._verify_helper is not None:
        self._context._verify_helper.raise_if_problem()
    if self._context._alpn_select_helper is not None:
        self._context._alpn_select_helper.raise_if_problem()
    if self._context._ocsp_helper is not None:
        self._context._ocsp_helper.raise_if_problem()
    error = _lib.SSL_get_error(ssl, result)
    if error == _lib.SSL_ERROR_WANT_READ:
        raise WantReadError()
    elif error == _lib.SSL_ERROR_WANT_WRITE:
        raise WantWriteError()
    elif error == _lib.SSL_ERROR_ZERO_RETURN:
        raise ZeroReturnError()
    elif error == _lib.SSL_ERROR_WANT_X509_LOOKUP:
        raise WantX509LookupError()
    elif error == _lib.SSL_ERROR_SYSCALL:
        if _lib.ERR_peek_error() == 0:
            if result < 0:
                if platform == 'win32':
                    errno = _ffi.getwinerror()[0]
                else:
                    errno = _ffi.errno
                if errno != 0:
                    raise SysCallError(errno, errorcode.get(errno))
            raise SysCallError(-1, 'Unexpected EOF')
        else:
            _raise_current_error()
    elif error == _lib.SSL_ERROR_SSL and _lib.ERR_peek_error() != 0:
        peeked_error = _lib.ERR_peek_error()
        reason = _lib.ERR_GET_REASON(peeked_error)
        if _lib.Cryptography_HAS_UNEXPECTED_EOF_WHILE_READING:
            _openssl_assert(reason == _lib.SSL_R_UNEXPECTED_EOF_WHILE_READING)
            _lib.ERR_clear_error()
            raise SysCallError(-1, 'Unexpected EOF')
        else:
            _raise_current_error()
    elif error == _lib.SSL_ERROR_NONE:
        pass
    else:
        _raise_current_error()