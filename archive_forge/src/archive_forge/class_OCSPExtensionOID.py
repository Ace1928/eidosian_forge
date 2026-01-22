from __future__ import annotations
import typing
from cryptography.hazmat.bindings._rust import (
from cryptography.hazmat.primitives import hashes
class OCSPExtensionOID:
    NONCE = ObjectIdentifier('1.3.6.1.5.5.7.48.1.2')
    ACCEPTABLE_RESPONSES = ObjectIdentifier('1.3.6.1.5.5.7.48.1.4')