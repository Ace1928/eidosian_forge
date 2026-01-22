from __future__ import annotations
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

    Set the named curve flag on the EC_KEY. This causes OpenSSL to
    serialize EC keys along with their curve OID which makes
    deserialization easier.
    