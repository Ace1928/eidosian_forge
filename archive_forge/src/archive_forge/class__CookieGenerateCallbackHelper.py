import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
class _CookieGenerateCallbackHelper(_CallbackExceptionHelper):

    def __init__(self, callback):
        _CallbackExceptionHelper.__init__(self)

        @wraps(callback)
        def wrapper(ssl, out, outlen):
            try:
                conn = Connection._reverse_mapping[ssl]
                cookie = callback(conn)
                out[0:len(cookie)] = cookie
                outlen[0] = len(cookie)
                return 1
            except Exception as e:
                self._problems.append(e)
                return 0
        self.callback = _ffi.callback('int (*)(SSL *, unsigned char *, unsigned int *)', wrapper)