import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def use_privatekey(self, pkey):
    """
        Load a private key from a PKey object

        :param pkey: The PKey object
        :return: None
        """
    if not isinstance(pkey, PKey):
        raise TypeError('pkey must be a PKey instance')
    use_result = _lib.SSL_use_PrivateKey(self._ssl, pkey._pkey)
    if not use_result:
        self._context._raise_passphrase_exception()