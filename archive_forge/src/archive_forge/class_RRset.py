import dns.name
import dns.rdataset
import dns.rdataclass
import dns.renderer
from ._compat import string_types
class RRset(dns.rdataset.Rdataset):
    """A DNS RRset (named rdataset).

    RRset inherits from Rdataset, and RRsets can be treated as
    Rdatasets in most cases.  There are, however, a few notable
    exceptions.  RRsets have different to_wire() and to_text() method
    arguments, reflecting the fact that RRsets always have an owner
    name.
    """
    __slots__ = ['name', 'deleting']

    def __init__(self, name, rdclass, rdtype, covers=dns.rdatatype.NONE, deleting=None):
        """Create a new RRset."""
        super(RRset, self).__init__(rdclass, rdtype, covers)
        self.name = name
        self.deleting = deleting

    def _clone(self):
        obj = super(RRset, self)._clone()
        obj.name = self.name
        obj.deleting = self.deleting
        return obj

    def __repr__(self):
        if self.covers == 0:
            ctext = ''
        else:
            ctext = '(' + dns.rdatatype.to_text(self.covers) + ')'
        if self.deleting is not None:
            dtext = ' delete=' + dns.rdataclass.to_text(self.deleting)
        else:
            dtext = ''
        return '<DNS ' + str(self.name) + ' ' + dns.rdataclass.to_text(self.rdclass) + ' ' + dns.rdatatype.to_text(self.rdtype) + ctext + dtext + ' RRset>'

    def __str__(self):
        return self.to_text()

    def __eq__(self, other):
        if not isinstance(other, RRset):
            return False
        if self.name != other.name:
            return False
        return super(RRset, self).__eq__(other)

    def match(self, name, rdclass, rdtype, covers, deleting=None):
        """Returns ``True`` if this rrset matches the specified class, type,
        covers, and deletion state.
        """
        if not super(RRset, self).match(rdclass, rdtype, covers):
            return False
        if self.name != name or self.deleting != deleting:
            return False
        return True

    def to_text(self, origin=None, relativize=True, **kw):
        """Convert the RRset into DNS master file format.

        See ``dns.name.Name.choose_relativity`` for more information
        on how *origin* and *relativize* determine the way names
        are emitted.

        Any additional keyword arguments are passed on to the rdata
        ``to_text()`` method.

        *origin*, a ``dns.name.Name`` or ``None``, the origin for relative
        names.

        *relativize*, a ``bool``.  If ``True``, names will be relativized
        to *origin*.
        """
        return super(RRset, self).to_text(self.name, origin, relativize, self.deleting, **kw)

    def to_wire(self, file, compress=None, origin=None, **kw):
        """Convert the RRset to wire format.

        All keyword arguments are passed to ``dns.rdataset.to_wire()``; see
        that function for details.

        Returns an ``int``, the number of records emitted.
        """
        return super(RRset, self).to_wire(self.name, file, compress, origin, self.deleting, **kw)

    def to_rdataset(self):
        """Convert an RRset into an Rdataset.

        Returns a ``dns.rdataset.Rdataset``.
        """
        return dns.rdataset.from_rdata_list(self.ttl, list(self))