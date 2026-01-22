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
def _header_line(self, section):
    """Process one line from the text format header section."""
    token = self.tok.get()
    what = token.value
    if what == 'id':
        self.message.id = self.tok.get_int()
    elif what == 'flags':
        while True:
            token = self.tok.get()
            if not token.is_identifier():
                self.tok.unget(token)
                break
            self.message.flags = self.message.flags | dns.flags.from_text(token.value)
        if dns.opcode.is_update(self.message.flags):
            self.updating = True
    elif what == 'edns':
        self.message.edns = self.tok.get_int()
        self.message.ednsflags = self.message.ednsflags | self.message.edns << 16
    elif what == 'eflags':
        if self.message.edns < 0:
            self.message.edns = 0
        while True:
            token = self.tok.get()
            if not token.is_identifier():
                self.tok.unget(token)
                break
            self.message.ednsflags = self.message.ednsflags | dns.flags.edns_from_text(token.value)
    elif what == 'payload':
        self.message.payload = self.tok.get_int()
        if self.message.edns < 0:
            self.message.edns = 0
    elif what == 'opcode':
        text = self.tok.get_string()
        self.message.flags = self.message.flags | dns.opcode.to_flags(dns.opcode.from_text(text))
    elif what == 'rcode':
        text = self.tok.get_string()
        self.message.set_rcode(dns.rcode.from_text(text))
    else:
        raise UnknownHeaderField
    self.tok.get_eol()