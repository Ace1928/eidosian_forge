from io import BytesIO
import struct
import random
import time
import dns.exception
import dns.tsig
from ._compat import long
def add_rrset(self, section, rrset, **kw):
    """Add the rrset to the specified section.

        Any keyword arguments are passed on to the rdataset's to_wire()
        routine.
        """
    self._set_section(section)
    before = self.output.tell()
    n = rrset.to_wire(self.output, self.compress, self.origin, **kw)
    after = self.output.tell()
    if after >= self.max_size:
        self._rollback(before)
        raise dns.exception.TooBig
    self.counts[section] += n