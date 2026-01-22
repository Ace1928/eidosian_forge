import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def DTLSv1_handle_timeout(self):
    """
        Handles any timeout events which have become pending on a DTLS SSL
        object.

        :return: `True` if there was a pending timeout, `False` otherwise.
        """
    result = _lib.DTLSv1_handle_timeout(self._ssl)
    if result < 0:
        self._raise_ssl_error(self._ssl, result)
    else:
        return bool(result)