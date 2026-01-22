from io import BytesIO
import struct
import random
import time
import dns.exception
import dns.tsig
from ._compat import long
def add_question(self, qname, rdtype, rdclass=dns.rdataclass.IN):
    """Add a question to the message."""
    self._set_section(QUESTION)
    before = self.output.tell()
    qname.to_wire(self.output, self.compress, self.origin)
    self.output.write(struct.pack('!HH', rdtype, rdclass))
    after = self.output.tell()
    if after >= self.max_size:
        self._rollback(before)
        raise dns.exception.TooBig
    self.counts[QUESTION] += 1