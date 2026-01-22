import dns.message
import dns.name
import dns.opcode
import dns.rdata
import dns.rdataclass
import dns.rdataset
import dns.tsig
from ._compat import string_types
def absent(self, name, rdtype=None):
    """Require that an owner name (and optionally an rdata type) does
        not exist as a prerequisite to the execution of the update."""
    if isinstance(name, string_types):
        name = dns.name.from_text(name, None)
    if rdtype is None:
        self.find_rrset(self.answer, name, dns.rdataclass.NONE, dns.rdatatype.ANY, dns.rdatatype.NONE, None, True, True)
    else:
        if isinstance(rdtype, string_types):
            rdtype = dns.rdatatype.from_text(rdtype)
        self.find_rrset(self.answer, name, dns.rdataclass.NONE, rdtype, dns.rdatatype.NONE, None, True, True)