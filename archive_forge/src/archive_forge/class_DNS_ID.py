from __future__ import annotations
import ipaddress
import re
from typing import Protocol, Sequence, Union, runtime_checkable
import attr
from .exceptions import (
@attr.s(init=False, slots=True)
class DNS_ID:
    """
    A DNS service ID, aka hostname.
    """
    hostname: bytes = attr.ib()
    _RE_LEGAL_CHARS = re.compile(b'^[a-z0-9\\-_.]+$')
    pattern_class = DNSPattern
    error_on_mismatch = DNSMismatch

    def __init__(self, hostname: str):
        if not isinstance(hostname, str):
            raise TypeError('DNS-ID must be a text string.')
        hostname = hostname.strip()
        if not hostname or _is_ip_address(hostname):
            raise ValueError('Invalid DNS-ID.')
        if any((ord(c) > 127 for c in hostname)):
            if idna:
                ascii_id = idna.encode(hostname)
            else:
                raise ImportError('idna library is required for non-ASCII IDs.')
        else:
            ascii_id = hostname.encode('ascii')
        self.hostname = ascii_id.translate(_TRANS_TO_LOWER)
        if self._RE_LEGAL_CHARS.match(self.hostname) is None:
            raise ValueError('Invalid DNS-ID.')

    def verify(self, pattern: CertificatePattern) -> bool:
        """
        https://tools.ietf.org/search/rfc6125#section-6.4
        """
        if isinstance(pattern, self.pattern_class):
            return _hostname_matches(pattern.pattern, self.hostname)
        return False