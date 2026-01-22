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
@dns.immutable.immutable
class ImmutableVersion(Version):

    def __init__(self, version: WritableVersion):
        super().__init__(version.zone, True)
        self.id = version.id
        self.origin = version.origin
        for name in version.changed:
            node = version.nodes.get(name)
            if node:
                version.nodes[name] = ImmutableVersionedNode(node)
        self.nodes = dns.immutable.Dict(version.nodes, True, self.zone.map_factory)