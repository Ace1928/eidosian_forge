from __future__ import annotations
import ipaddress
import re
from typing import Protocol, Sequence, Union, runtime_checkable
import attr
from .exceptions import (
@attr.s(slots=True)
class ServiceMatch:
    """
    A match of a service id and a certificate pattern.
    """
    service_id: ServiceID = attr.ib()
    cert_pattern: CertificatePattern = attr.ib()