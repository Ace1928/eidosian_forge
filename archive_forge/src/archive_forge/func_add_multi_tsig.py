from io import BytesIO
import struct
import random
import time
import dns.exception
import dns.tsig
from ._compat import long
def add_multi_tsig(self, ctx, keyname, secret, fudge, id, tsig_error, other_data, request_mac, algorithm=dns.tsig.default_algorithm):
    """Add a TSIG signature to the message. Unlike add_tsig(), this can be
        used for a series of consecutive DNS envelopes, e.g. for a zone
        transfer over TCP [RFC2845, 4.4].

        For the first message in the sequence, give ctx=None. For each
        subsequent message, give the ctx that was returned from the
        add_multi_tsig() call for the previous message."""
    s = self.output.getvalue()
    tsig_rdata, self.mac, ctx = dns.tsig.sign(s, keyname, secret, int(time.time()), fudge, id, tsig_error, other_data, request_mac, ctx=ctx, first=ctx is None, multi=True, algorithm=algorithm)
    self._write_tsig(tsig_rdata, keyname)
    return ctx