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
def _bio_to_string(bio: Any) -> bytes:
    """
    Copy the contents of an OpenSSL BIO object into a Python byte string.
    """
    result_buffer = _ffi.new('char**')
    buffer_length = _lib.BIO_get_mem_data(bio, result_buffer)
    return _ffi.buffer(result_buffer[0], buffer_length)[:]