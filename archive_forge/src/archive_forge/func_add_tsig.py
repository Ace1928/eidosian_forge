from io import BytesIO
import struct
import random
import time
import dns.exception
import dns.tsig
from ._compat import long
def add_tsig(self, keyname, secret, fudge, id, tsig_error, other_data, request_mac, algorithm=dns.tsig.default_algorithm):
    """Add a TSIG signature to the message."""
    s = self.output.getvalue()
    tsig_rdata, self.mac, ctx = dns.tsig.sign(s, keyname, secret, int(time.time()), fudge, id, tsig_error, other_data, request_mac, algorithm=algorithm)
    self._write_tsig(tsig_rdata, keyname)