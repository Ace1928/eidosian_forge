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
def to_cryptography(self) -> x509.CertificateRevocationList:
    """
        Export as a ``cryptography`` CRL.

        :rtype: ``cryptography.x509.CertificateRevocationList``

        .. versionadded:: 17.1.0
        """
    from cryptography.x509 import load_der_x509_crl
    der = dump_crl(FILETYPE_ASN1, self)
    return load_der_x509_crl(der)