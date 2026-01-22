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
@classmethod
def _from_raw_x509_ptr(cls, x509: Any) -> 'X509':
    cert = cls.__new__(cls)
    cert._x509 = _ffi.gc(x509, _lib.X509_free)
    cert._issuer_invalidator = _X509NameInvalidator()
    cert._subject_invalidator = _X509NameInvalidator()
    return cert