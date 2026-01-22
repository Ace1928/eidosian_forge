import contextlib
import io
import os
import struct
from typing import (
import dns.exception
import dns.grange
import dns.immutable
import dns.name
import dns.node
import dns.rdata
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rdtypes.ANY.SOA
import dns.rdtypes.ANY.ZONEMD
import dns.rrset
import dns.tokenizer
import dns.transaction
import dns.ttl
import dns.zonefile
from dns.zonetypes import DigestHashAlgorithm, DigestScheme, _digest_hashers
def _origin_information(self):
    absolute, relativize, effective = self.manager.origin_information()
    if absolute is None and self.version.origin is not None:
        absolute = self.version.origin
        if relativize:
            effective = dns.name.empty
        else:
            effective = absolute
    return (absolute, relativize, effective)