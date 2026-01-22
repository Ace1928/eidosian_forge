import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def _handle_bio_errors(self, bio, result):
    if _lib.BIO_should_retry(bio):
        if _lib.BIO_should_read(bio):
            raise WantReadError()
        elif _lib.BIO_should_write(bio):
            raise WantWriteError()
        elif _lib.BIO_should_io_special(bio):
            raise ValueError('BIO_should_io_special')
        else:
            raise ValueError('unknown bio failure')
    else:
        _raise_current_error()