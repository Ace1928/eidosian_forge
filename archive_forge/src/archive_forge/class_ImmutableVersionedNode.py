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
class ImmutableVersionedNode(VersionedNode):

    def __init__(self, node):
        super().__init__()
        self.id = node.id
        self.rdatasets = tuple([dns.rdataset.ImmutableRdataset(rds) for rds in node.rdatasets])

    def find_rdataset(self, rdclass: dns.rdataclass.RdataClass, rdtype: dns.rdatatype.RdataType, covers: dns.rdatatype.RdataType=dns.rdatatype.NONE, create: bool=False) -> dns.rdataset.Rdataset:
        if create:
            raise TypeError('immutable')
        return super().find_rdataset(rdclass, rdtype, covers, False)

    def get_rdataset(self, rdclass: dns.rdataclass.RdataClass, rdtype: dns.rdatatype.RdataType, covers: dns.rdatatype.RdataType=dns.rdatatype.NONE, create: bool=False) -> Optional[dns.rdataset.Rdataset]:
        if create:
            raise TypeError('immutable')
        return super().get_rdataset(rdclass, rdtype, covers, False)

    def delete_rdataset(self, rdclass: dns.rdataclass.RdataClass, rdtype: dns.rdatatype.RdataType, covers: dns.rdatatype.RdataType=dns.rdatatype.NONE) -> None:
        raise TypeError('immutable')

    def replace_rdataset(self, replacement: dns.rdataset.Rdataset) -> None:
        raise TypeError('immutable')

    def is_immutable(self) -> bool:
        return True