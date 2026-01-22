import calendar
import datetime
import functools
from base64 import b16encode
from functools import partial
from os import PathLike
from typing import (
from cryptography import utils, x509
from cryptography.hazmat.primitives.asymmetric import (
from OpenSSL._util import (
def _new_mem_buf(buffer: Optional[bytes]=None) -> Any:
    """
    Allocate a new OpenSSL memory BIO.

    Arrange for the garbage collector to clean it up automatically.

    :param buffer: None or some bytes to use to put into the BIO so that they
        can be read out.
    """
    if buffer is None:
        bio = _lib.BIO_new(_lib.BIO_s_mem())
        free = _lib.BIO_free
    else:
        data = _ffi.new('char[]', buffer)
        bio = _lib.BIO_new_mem_buf(data, len(buffer))

        def free(bio: Any, ref: Any=data) -> Any:
            return _lib.BIO_free(bio)
    _openssl_assert(bio != _ffi.NULL)
    bio = _ffi.gc(bio, free)
    return bio