import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def use_privatekey_file(self, keyfile, filetype=_UNSPECIFIED):
    """
        Load a private key from a file

        :param keyfile: The name of the key file (``bytes`` or ``unicode``)
        :param filetype: (optional) The encoding of the file, which is either
            :const:`FILETYPE_PEM` or :const:`FILETYPE_ASN1`.  The default is
            :const:`FILETYPE_PEM`.

        :return: None
        """
    keyfile = _path_bytes(keyfile)
    if filetype is _UNSPECIFIED:
        filetype = FILETYPE_PEM
    elif not isinstance(filetype, int):
        raise TypeError('filetype must be an integer')
    use_result = _lib.SSL_CTX_use_PrivateKey_file(self._context, keyfile, filetype)
    if not use_result:
        self._raise_passphrase_exception()