from __future__ import generators
import sys
import re
import os
from io import BytesIO
import dns.exception
import dns.name
import dns.node
import dns.rdataclass
import dns.rdatatype
import dns.rdata
import dns.rdtypes.ANY.SOA
import dns.rrset
import dns.tokenizer
import dns.ttl
import dns.grange
from ._compat import string_types, text_type, PY3
def _validate_name(self, name):
    if isinstance(name, string_types):
        name = dns.name.from_text(name, None)
    elif not isinstance(name, dns.name.Name):
        raise KeyError('name parameter must be convertible to a DNS name')
    if name.is_absolute():
        if not name.is_subdomain(self.origin):
            raise KeyError('name parameter must be a subdomain of the zone origin')
        if self.relativize:
            name = name.relativize(self.origin)
    return name