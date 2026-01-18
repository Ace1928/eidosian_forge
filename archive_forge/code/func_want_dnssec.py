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
def want_dnssec(self, wanted=True):
    """Enable or disable 'DNSSEC desired' flag in requests.

        *wanted*, a ``bool``.  If ``True``, then DNSSEC data is
        desired in the response, EDNS is enabled if required, and then
        the DO bit is set.  If ``False``, the DO bit is cleared if
        EDNS is enabled.
        """
    if wanted:
        if self.edns < 0:
            self.use_edns()
        self.ednsflags |= dns.flags.DO
    elif self.edns >= 0:
        self.ednsflags &= ~dns.flags.DO