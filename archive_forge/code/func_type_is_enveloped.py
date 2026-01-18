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
def type_is_enveloped(self) -> bool:
    """
        Check if this NID_pkcs7_enveloped object

        :returns: True if the PKCS7 is of type enveloped
        """
    return bool(_lib.PKCS7_type_is_enveloped(self._pkcs7))