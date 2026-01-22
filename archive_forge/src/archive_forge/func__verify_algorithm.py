from __future__ import annotations
import abc
import datetime
import typing
from cryptography import utils, x509
from cryptography.hazmat.bindings._rust import ocsp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.base import (
def _verify_algorithm(algorithm: hashes.HashAlgorithm) -> None:
    if not isinstance(algorithm, _ALLOWED_HASHES):
        raise ValueError('Algorithm must be SHA1, SHA224, SHA256, SHA384, or SHA512')