from __future__ import annotations
import typing
from cryptography.hazmat.bindings._rust import (
from cryptography.hazmat.primitives import hashes
class CertificatePoliciesOID:
    CPS_QUALIFIER = ObjectIdentifier('1.3.6.1.5.5.7.2.1')
    CPS_USER_NOTICE = ObjectIdentifier('1.3.6.1.5.5.7.2.2')
    ANY_POLICY = ObjectIdentifier('2.5.29.32.0')