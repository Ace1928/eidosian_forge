from __future__ import absolute_import
from io import StringIO
import struct
import time
import dns.edns
import dns.exception
import dns.flags
import dns.name
import dns.opcode
import dns.entropy
import dns.rcode
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.rrset
import dns.renderer
import dns.tsig
import dns.wiredata
from ._compat import long, xrange, string_types
def _get_question(self, qcount):
    """Read the next *qcount* records from the wire data and add them to
        the question section.
        """
    if self.updating and qcount > 1:
        raise dns.exception.FormError
    for i in xrange(0, qcount):
        qname, used = dns.name.from_wire(self.wire, self.current)
        if self.message.origin is not None:
            qname = qname.relativize(self.message.origin)
        self.current = self.current + used
        rdtype, rdclass = struct.unpack('!HH', self.wire[self.current:self.current + 4])
        self.current = self.current + 4
        self.message.find_rrset(self.message.question, qname, rdclass, rdtype, create=True, force_unique=True)
        if self.updating:
            self.zone_rdclass = rdclass