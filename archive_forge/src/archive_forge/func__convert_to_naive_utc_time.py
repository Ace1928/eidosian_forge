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
def _convert_to_naive_utc_time(time: datetime.datetime) -> datetime.datetime:
    """Normalizes a datetime to a naive datetime in UTC.

    time -- datetime to normalize. Assumed to be in UTC if not timezone
            aware.
    """
    if time.tzinfo is not None:
        offset = time.utcoffset()
        offset = offset if offset else datetime.timedelta()
        return time.replace(tzinfo=None) - offset
    else:
        return time