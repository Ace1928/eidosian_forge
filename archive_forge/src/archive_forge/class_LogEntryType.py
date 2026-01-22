from __future__ import annotations
import abc
import datetime
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.hazmat.primitives.hashes import HashAlgorithm
class LogEntryType(utils.Enum):
    X509_CERTIFICATE = 0
    PRE_CERTIFICATE = 1