from __future__ import annotations
import binascii
import re
import sys
import typing
import warnings
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.x509.oid import NameOID, ObjectIdentifier
def _read_re(self, pat) -> str:
    match = pat.match(self._data, pos=self._idx)
    if match is None:
        raise ValueError
    val = match.group()
    self._idx += len(val)
    return val