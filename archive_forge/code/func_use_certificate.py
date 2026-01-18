import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def use_certificate(self, cert):
    """
        Load a certificate from a X509 object

        :param cert: The X509 object
        :return: None
        """
    if not isinstance(cert, X509):
        raise TypeError('cert must be an X509 instance')
    use_result = _lib.SSL_use_certificate(self._ssl, cert._x509)
    if not use_result:
        _raise_current_error()