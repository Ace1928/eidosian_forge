import contextlib
import io
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import dns.edns
import dns.entropy
import dns.enum
import dns.exception
import dns.flags
import dns.name
import dns.opcode
import dns.rcode
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.rdtypes.ANY.OPT
import dns.rdtypes.ANY.TSIG
import dns.renderer
import dns.rrset
import dns.tsig
import dns.ttl
import dns.wire
@staticmethod
def _make_opt(flags=0, payload=DEFAULT_EDNS_PAYLOAD, options=None):
    opt = dns.rdtypes.ANY.OPT.OPT(payload, dns.rdatatype.OPT, options or ())
    return dns.rrset.from_rdata(dns.name.root, int(flags), opt)