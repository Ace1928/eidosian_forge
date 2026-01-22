import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
class _OCSPClientCallbackHelper(_CallbackExceptionHelper):
    """
    Wrap a callback such that it can be used as an OCSP callback for the client
    side.

    Annoyingly, OpenSSL defines one OCSP callback but uses it in two different
    ways. For servers, that callback is expected to retrieve some OCSP data and
    hand it to OpenSSL, and may return only SSL_TLSEXT_ERR_OK,
    SSL_TLSEXT_ERR_FATAL, and SSL_TLSEXT_ERR_NOACK. For clients, that callback
    is expected to check the OCSP data, and returns a negative value on error,
    0 if the response is not acceptable, or positive if it is. These are
    mutually exclusive return code behaviours, and they mean that we need two
    helpers so that we always return an appropriate error code if the user's
    code throws an exception.

    Given that we have to have two helpers anyway, these helpers are a bit more
    helpery than most: specifically, they hide a few more of the OpenSSL
    functions so that the user has an easier time writing these callbacks.

    This helper implements the client side.
    """

    def __init__(self, callback):
        _CallbackExceptionHelper.__init__(self)

        @wraps(callback)
        def wrapper(ssl, cdata):
            try:
                conn = Connection._reverse_mapping[ssl]
                if cdata != _ffi.NULL:
                    data = _ffi.from_handle(cdata)
                else:
                    data = None
                ocsp_ptr = _ffi.new('unsigned char **')
                ocsp_len = _lib.SSL_get_tlsext_status_ocsp_resp(ssl, ocsp_ptr)
                if ocsp_len < 0:
                    ocsp_data = b''
                else:
                    ocsp_data = _ffi.buffer(ocsp_ptr[0], ocsp_len)[:]
                valid = callback(conn, ocsp_data, data)
                return int(bool(valid))
            except Exception as e:
                self._problems.append(e)
                return -1
        self.callback = _ffi.callback('int (*)(SSL *, void *)', wrapper)