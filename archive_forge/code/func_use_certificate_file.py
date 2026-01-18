import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def use_certificate_file(self, certfile, filetype=FILETYPE_PEM):
    """
        Load a certificate from a file

        :param certfile: The name of the certificate file (``bytes`` or
            ``unicode``).
        :param filetype: (optional) The encoding of the file, which is either
            :const:`FILETYPE_PEM` or :const:`FILETYPE_ASN1`.  The default is
            :const:`FILETYPE_PEM`.

        :return: None
        """
    certfile = _path_bytes(certfile)
    if not isinstance(filetype, int):
        raise TypeError('filetype must be an integer')
    use_result = _lib.SSL_CTX_use_certificate_file(self._context, certfile, filetype)
    if not use_result:
        _raise_current_error()