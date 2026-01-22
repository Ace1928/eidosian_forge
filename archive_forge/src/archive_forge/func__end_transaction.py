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
def _end_transaction(self, commit):
    if commit and self._changed():
        rrsets = []
        for (name, _, _), rdataset in self.rdatasets.items():
            rrset = dns.rrset.RRset(name, rdataset.rdclass, rdataset.rdtype, rdataset.covers)
            rrset.update(rdataset)
            rrsets.append(rrset)
        self.manager.set_rrsets(rrsets)