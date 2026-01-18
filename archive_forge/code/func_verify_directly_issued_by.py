from __future__ import annotations
import abc
import datetime
import os
import typing
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.extensions import (
from cryptography.x509.name import Name, _ASN1Type
from cryptography.x509.oid import ObjectIdentifier
@abc.abstractmethod
def verify_directly_issued_by(self, issuer: Certificate) -> None:
    """
        This method verifies that certificate issuer name matches the
        issuer subject name and that the certificate is signed by the
        issuer's private key. No other validation is performed.
        """