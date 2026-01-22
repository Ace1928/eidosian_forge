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
class _WireReader(object):
    """Wire format reader.

    wire: a binary, is the wire-format message.
    message: The message object being built
    current: When building a message object from wire format, this
    variable contains the offset from the beginning of wire of the next octet
    to be read.
    updating: Is the message a dynamic update?
    one_rr_per_rrset: Put each RR into its own RRset?
    ignore_trailing: Ignore trailing junk at end of request?
    zone_rdclass: The class of the zone in messages which are
    DNS dynamic updates.
    """

    def __init__(self, wire, message, question_only=False, one_rr_per_rrset=False, ignore_trailing=False):
        self.wire = dns.wiredata.maybe_wrap(wire)
        self.message = message
        self.current = 0
        self.updating = False
        self.zone_rdclass = dns.rdataclass.IN
        self.question_only = question_only
        self.one_rr_per_rrset = one_rr_per_rrset
        self.ignore_trailing = ignore_trailing

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

    def _get_section(self, section, count):
        """Read the next I{count} records from the wire data and add them to
        the specified section.

        section: the section of the message to which to add records
        count: the number of records to read
        """
        if self.updating or self.one_rr_per_rrset:
            force_unique = True
        else:
            force_unique = False
        seen_opt = False
        for i in xrange(0, count):
            rr_start = self.current
            name, used = dns.name.from_wire(self.wire, self.current)
            absolute_name = name
            if self.message.origin is not None:
                name = name.relativize(self.message.origin)
            self.current = self.current + used
            rdtype, rdclass, ttl, rdlen = struct.unpack('!HHIH', self.wire[self.current:self.current + 10])
            self.current = self.current + 10
            if rdtype == dns.rdatatype.OPT:
                if section is not self.message.additional or seen_opt:
                    raise BadEDNS
                self.message.payload = rdclass
                self.message.ednsflags = ttl
                self.message.edns = (ttl & 16711680) >> 16
                self.message.options = []
                current = self.current
                optslen = rdlen
                while optslen > 0:
                    otype, olen = struct.unpack('!HH', self.wire[current:current + 4])
                    current = current + 4
                    opt = dns.edns.option_from_wire(otype, self.wire, current, olen)
                    self.message.options.append(opt)
                    current = current + olen
                    optslen = optslen - 4 - olen
                seen_opt = True
            elif rdtype == dns.rdatatype.TSIG:
                if not (section is self.message.additional and i == count - 1):
                    raise BadTSIG
                if self.message.keyring is None:
                    raise UnknownTSIGKey('got signed message without keyring')
                secret = self.message.keyring.get(absolute_name)
                if secret is None:
                    raise UnknownTSIGKey("key '%s' unknown" % name)
                self.message.keyname = absolute_name
                self.message.keyalgorithm, self.message.mac = dns.tsig.get_algorithm_and_mac(self.wire, self.current, rdlen)
                self.message.tsig_ctx = dns.tsig.validate(self.wire, absolute_name, secret, int(time.time()), self.message.request_mac, rr_start, self.current, rdlen, self.message.tsig_ctx, self.message.multi, self.message.first)
                self.message.had_tsig = True
            else:
                if ttl < 0:
                    ttl = 0
                if self.updating and (rdclass == dns.rdataclass.ANY or rdclass == dns.rdataclass.NONE):
                    deleting = rdclass
                    rdclass = self.zone_rdclass
                else:
                    deleting = None
                if deleting == dns.rdataclass.ANY or (deleting == dns.rdataclass.NONE and section is self.message.answer):
                    covers = dns.rdatatype.NONE
                    rd = None
                else:
                    rd = dns.rdata.from_wire(rdclass, rdtype, self.wire, self.current, rdlen, self.message.origin)
                    covers = rd.covers()
                if self.message.xfr and rdtype == dns.rdatatype.SOA:
                    force_unique = True
                rrset = self.message.find_rrset(section, name, rdclass, rdtype, covers, deleting, True, force_unique)
                if rd is not None:
                    rrset.add(rd, ttl)
            self.current = self.current + rdlen

    def read(self):
        """Read a wire format DNS message and build a dns.message.Message
        object."""
        l = len(self.wire)
        if l < 12:
            raise ShortHeader
        self.message.id, self.message.flags, qcount, ancount, aucount, adcount = struct.unpack('!HHHHHH', self.wire[:12])
        self.current = 12
        if dns.opcode.is_update(self.message.flags):
            self.updating = True
        self._get_question(qcount)
        if self.question_only:
            return
        self._get_section(self.message.answer, ancount)
        self._get_section(self.message.authority, aucount)
        self._get_section(self.message.additional, adcount)
        if not self.ignore_trailing and self.current != l:
            raise TrailingJunk
        if self.message.multi and self.message.tsig_ctx and (not self.message.had_tsig):
            self.message.tsig_ctx.update(self.wire)