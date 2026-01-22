import contextlib
import io
import random
import struct
import time
import dns.exception
import dns.tsig
def add_opt(self, opt, pad=0, opt_size=0, tsig_size=0):
    """Add *opt* to the additional section, applying padding if desired.  The
        padding will take the specified precomputed OPT size and TSIG size into
        account.

        Note that we don't have reliable way of knowing how big a GSS-TSIG digest
        might be, so we we might not get an even multiple of the pad in that case."""
    if pad:
        ttl = opt.ttl
        assert opt_size >= 11
        opt_rdata = opt[0]
        size_without_padding = self.output.tell() + opt_size + tsig_size
        remainder = size_without_padding % pad
        if remainder:
            pad = b'\x00' * (pad - remainder)
        else:
            pad = b''
        options = list(opt_rdata.options)
        options.append(dns.edns.GenericOption(dns.edns.OptionType.PADDING, pad))
        opt = dns.message.Message._make_opt(ttl, opt_rdata.rdclass, options)
        self.was_padded = True
    self.add_rrset(ADDITIONAL, opt)