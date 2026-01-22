from io import BytesIO
import base64
import binascii
import dns.exception
import dns.name
import dns.rdataclass
import dns.rdatatype
import dns.tokenizer
import dns.wiredata
from ._compat import xrange, string_types, text_type
class Rdata(object):
    """Base class for all DNS rdata types."""
    __slots__ = ['rdclass', 'rdtype']

    def __init__(self, rdclass, rdtype):
        """Initialize an rdata.

        *rdclass*, an ``int`` is the rdataclass of the Rdata.
        *rdtype*, an ``int`` is the rdatatype of the Rdata.
        """
        self.rdclass = rdclass
        self.rdtype = rdtype

    def covers(self):
        """Return the type a Rdata covers.

        DNS SIG/RRSIG rdatas apply to a specific type; this type is
        returned by the covers() function.  If the rdata type is not
        SIG or RRSIG, dns.rdatatype.NONE is returned.  This is useful when
        creating rdatasets, allowing the rdataset to contain only RRSIGs
        of a particular type, e.g. RRSIG(NS).

        Returns an ``int``.
        """
        return dns.rdatatype.NONE

    def extended_rdatatype(self):
        """Return a 32-bit type value, the least significant 16 bits of
        which are the ordinary DNS type, and the upper 16 bits of which are
        the "covered" type, if any.

        Returns an ``int``.
        """
        return self.covers() << 16 | self.rdtype

    def to_text(self, origin=None, relativize=True, **kw):
        """Convert an rdata to text format.

        Returns a ``text``.
        """
        raise NotImplementedError

    def to_wire(self, file, compress=None, origin=None):
        """Convert an rdata to wire format.

        Returns a ``binary``.
        """
        raise NotImplementedError

    def to_digestable(self, origin=None):
        """Convert rdata to a format suitable for digesting in hashes.  This
        is also the DNSSEC canonical form.

        Returns a ``binary``.
        """
        f = BytesIO()
        self.to_wire(f, None, origin)
        return f.getvalue()

    def validate(self):
        """Check that the current contents of the rdata's fields are
        valid.

        If you change an rdata by assigning to its fields,
        it is a good idea to call validate() when you are done making
        changes.

        Raises various exceptions if there are problems.

        Returns ``None``.
        """
        dns.rdata.from_text(self.rdclass, self.rdtype, self.to_text())

    def __repr__(self):
        covers = self.covers()
        if covers == dns.rdatatype.NONE:
            ctext = ''
        else:
            ctext = '(' + dns.rdatatype.to_text(covers) + ')'
        return '<DNS ' + dns.rdataclass.to_text(self.rdclass) + ' ' + dns.rdatatype.to_text(self.rdtype) + ctext + ' rdata: ' + str(self) + '>'

    def __str__(self):
        return self.to_text()

    def _cmp(self, other):
        """Compare an rdata with another rdata of the same rdtype and
        rdclass.

        Return < 0 if self < other in the DNSSEC ordering, 0 if self
        == other, and > 0 if self > other.

        """
        our = self.to_digestable(dns.name.root)
        their = other.to_digestable(dns.name.root)
        if our == their:
            return 0
        elif our > their:
            return 1
        else:
            return -1

    def __eq__(self, other):
        if not isinstance(other, Rdata):
            return False
        if self.rdclass != other.rdclass or self.rdtype != other.rdtype:
            return False
        return self._cmp(other) == 0

    def __ne__(self, other):
        if not isinstance(other, Rdata):
            return True
        if self.rdclass != other.rdclass or self.rdtype != other.rdtype:
            return True
        return self._cmp(other) != 0

    def __lt__(self, other):
        if not isinstance(other, Rdata) or self.rdclass != other.rdclass or self.rdtype != other.rdtype:
            return NotImplemented
        return self._cmp(other) < 0

    def __le__(self, other):
        if not isinstance(other, Rdata) or self.rdclass != other.rdclass or self.rdtype != other.rdtype:
            return NotImplemented
        return self._cmp(other) <= 0

    def __ge__(self, other):
        if not isinstance(other, Rdata) or self.rdclass != other.rdclass or self.rdtype != other.rdtype:
            return NotImplemented
        return self._cmp(other) >= 0

    def __gt__(self, other):
        if not isinstance(other, Rdata) or self.rdclass != other.rdclass or self.rdtype != other.rdtype:
            return NotImplemented
        return self._cmp(other) > 0

    def __hash__(self):
        return hash(self.to_digestable(dns.name.root))

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True):
        raise NotImplementedError

    @classmethod
    def from_wire(cls, rdclass, rdtype, wire, current, rdlen, origin=None):
        raise NotImplementedError

    def choose_relativity(self, origin=None, relativize=True):
        """Convert any domain names in the rdata to the specified
        relativization.
        """