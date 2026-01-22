import re
import sys
from typing import Any, Iterable, List, Optional, Set, Tuple, Union
import dns.exception
import dns.grange
import dns.name
import dns.node
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.rdtypes.ANY.SOA
import dns.rrset
import dns.tokenizer
import dns.transaction
import dns.ttl
class RRSetsReaderManager(dns.transaction.TransactionManager):

    def __init__(self, origin=dns.name.root, relativize=False, rdclass=dns.rdataclass.IN):
        self.origin = origin
        self.relativize = relativize
        self.rdclass = rdclass
        self.rrsets = []

    def reader(self):
        raise NotImplementedError

    def writer(self, replacement=False):
        assert replacement is True
        return RRsetsReaderTransaction(self, True, False)

    def get_class(self):
        return self.rdclass

    def origin_information(self):
        if self.relativize:
            effective = dns.name.empty
        else:
            effective = self.origin
        return (self.origin, self.relativize, effective)

    def set_rrsets(self, rrsets):
        self.rrsets = rrsets