import random
from io import StringIO
import struct
import dns.exception
import dns.rdatatype
import dns.rdataclass
import dns.rdata
import dns.set
from ._compat import string_types
class Rdataset(dns.set.Set):
    """A DNS rdataset."""
    __slots__ = ['rdclass', 'rdtype', 'covers', 'ttl']

    def __init__(self, rdclass, rdtype, covers=dns.rdatatype.NONE, ttl=0):
        """Create a new rdataset of the specified class and type.

        *rdclass*, an ``int``, the rdataclass.

        *rdtype*, an ``int``, the rdatatype.

        *covers*, an ``int``, the covered rdatatype.

        *ttl*, an ``int``, the TTL.
        """
        super(Rdataset, self).__init__()
        self.rdclass = rdclass
        self.rdtype = rdtype
        self.covers = covers
        self.ttl = ttl

    def _clone(self):
        obj = super(Rdataset, self)._clone()
        obj.rdclass = self.rdclass
        obj.rdtype = self.rdtype
        obj.covers = self.covers
        obj.ttl = self.ttl
        return obj

    def update_ttl(self, ttl):
        """Perform TTL minimization.

        Set the TTL of the rdataset to be the lesser of the set's current
        TTL or the specified TTL.  If the set contains no rdatas, set the TTL
        to the specified TTL.

        *ttl*, an ``int``.
        """
        if len(self) == 0:
            self.ttl = ttl
        elif ttl < self.ttl:
            self.ttl = ttl

    def add(self, rd, ttl=None):
        """Add the specified rdata to the rdataset.

        If the optional *ttl* parameter is supplied, then
        ``self.update_ttl(ttl)`` will be called prior to adding the rdata.

        *rd*, a ``dns.rdata.Rdata``, the rdata

        *ttl*, an ``int``, the TTL.

        Raises ``dns.rdataset.IncompatibleTypes`` if the type and class
        do not match the type and class of the rdataset.

        Raises ``dns.rdataset.DifferingCovers`` if the type is a signature
        type and the covered type does not match that of the rdataset.
        """
        if self.rdclass != rd.rdclass or self.rdtype != rd.rdtype:
            raise IncompatibleTypes
        if ttl is not None:
            self.update_ttl(ttl)
        if self.rdtype == dns.rdatatype.RRSIG or self.rdtype == dns.rdatatype.SIG:
            covers = rd.covers()
            if len(self) == 0 and self.covers == dns.rdatatype.NONE:
                self.covers = covers
            elif self.covers != covers:
                raise DifferingCovers
        if dns.rdatatype.is_singleton(rd.rdtype) and len(self) > 0:
            self.clear()
        super(Rdataset, self).add(rd)

    def union_update(self, other):
        self.update_ttl(other.ttl)
        super(Rdataset, self).union_update(other)

    def intersection_update(self, other):
        self.update_ttl(other.ttl)
        super(Rdataset, self).intersection_update(other)

    def update(self, other):
        """Add all rdatas in other to self.

        *other*, a ``dns.rdataset.Rdataset``, the rdataset from which
        to update.
        """
        self.update_ttl(other.ttl)
        super(Rdataset, self).update(other)

    def __repr__(self):
        if self.covers == 0:
            ctext = ''
        else:
            ctext = '(' + dns.rdatatype.to_text(self.covers) + ')'
        return '<DNS ' + dns.rdataclass.to_text(self.rdclass) + ' ' + dns.rdatatype.to_text(self.rdtype) + ctext + ' rdataset>'

    def __str__(self):
        return self.to_text()

    def __eq__(self, other):
        if not isinstance(other, Rdataset):
            return False
        if self.rdclass != other.rdclass or self.rdtype != other.rdtype or self.covers != other.covers:
            return False
        return super(Rdataset, self).__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_text(self, name=None, origin=None, relativize=True, override_rdclass=None, **kw):
        """Convert the rdataset into DNS master file format.

        See ``dns.name.Name.choose_relativity`` for more information
        on how *origin* and *relativize* determine the way names
        are emitted.

        Any additional keyword arguments are passed on to the rdata
        ``to_text()`` method.

        *name*, a ``dns.name.Name``.  If name is not ``None``, emit RRs with
        *name* as the owner name.

        *origin*, a ``dns.name.Name`` or ``None``, the origin for relative
        names.

        *relativize*, a ``bool``.  If ``True``, names will be relativized
        to *origin*.
        """
        if name is not None:
            name = name.choose_relativity(origin, relativize)
            ntext = str(name)
            pad = ' '
        else:
            ntext = ''
            pad = ''
        s = StringIO()
        if override_rdclass is not None:
            rdclass = override_rdclass
        else:
            rdclass = self.rdclass
        if len(self) == 0:
            s.write(u'{}{}{} {}\n'.format(ntext, pad, dns.rdataclass.to_text(rdclass), dns.rdatatype.to_text(self.rdtype)))
        else:
            for rd in self:
                s.write(u'%s%s%d %s %s %s\n' % (ntext, pad, self.ttl, dns.rdataclass.to_text(rdclass), dns.rdatatype.to_text(self.rdtype), rd.to_text(origin=origin, relativize=relativize, **kw)))
        return s.getvalue()[:-1]

    def to_wire(self, name, file, compress=None, origin=None, override_rdclass=None, want_shuffle=True):
        """Convert the rdataset to wire format.

        *name*, a ``dns.name.Name`` is the owner name to use.

        *file* is the file where the name is emitted (typically a
        BytesIO file).

        *compress*, a ``dict``, is the compression table to use.  If
        ``None`` (the default), names will not be compressed.

        *origin* is a ``dns.name.Name`` or ``None``.  If the name is
        relative and origin is not ``None``, then *origin* will be appended
        to it.

        *override_rdclass*, an ``int``, is used as the class instead of the
        class of the rdataset.  This is useful when rendering rdatasets
        associated with dynamic updates.

        *want_shuffle*, a ``bool``.  If ``True``, then the order of the
        Rdatas within the Rdataset will be shuffled before rendering.

        Returns an ``int``, the number of records emitted.
        """
        if override_rdclass is not None:
            rdclass = override_rdclass
            want_shuffle = False
        else:
            rdclass = self.rdclass
        file.seek(0, 2)
        if len(self) == 0:
            name.to_wire(file, compress, origin)
            stuff = struct.pack('!HHIH', self.rdtype, rdclass, 0, 0)
            file.write(stuff)
            return 1
        else:
            if want_shuffle:
                l = list(self)
                random.shuffle(l)
            else:
                l = self
            for rd in l:
                name.to_wire(file, compress, origin)
                stuff = struct.pack('!HHIH', self.rdtype, rdclass, self.ttl, 0)
                file.write(stuff)
                start = file.tell()
                rd.to_wire(file, compress, origin)
                end = file.tell()
                assert end - start < 65536
                file.seek(start - 2)
                stuff = struct.pack('!H', end - start)
                file.write(stuff)
                file.seek(0, 2)
            return len(self)

    def match(self, rdclass, rdtype, covers):
        """Returns ``True`` if this rdataset matches the specified class,
        type, and covers.
        """
        if self.rdclass == rdclass and self.rdtype == rdtype and (self.covers == covers):
            return True
        return False