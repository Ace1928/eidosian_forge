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
@property
def _nid(self) -> Any:
    return _lib.OBJ_obj2nid(_lib.X509_EXTENSION_get_object(self._extension))