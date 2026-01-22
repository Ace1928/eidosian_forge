from __future__ import annotations
import typing
from cryptography.hazmat.bindings._rust import (
from cryptography.hazmat.primitives import hashes
class SubjectInformationAccessOID:
    CA_REPOSITORY = ObjectIdentifier('1.3.6.1.5.5.7.48.5')