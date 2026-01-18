import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def want_read(self):
    """
        Checks if more data has to be read from the transport layer to complete
        an operation.

        :return: True iff more data has to be read
        """
    return _lib.SSL_want_read(self._ssl)