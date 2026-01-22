import struct
import dns.exception
import dns.rdata
import dns.name
class PX(dns.rdata.Rdata):
    """PX record.

    @ivar preference: the preference value
    @type preference: int
    @ivar map822: the map822 name
    @type map822: dns.name.Name object
    @ivar mapx400: the mapx400 name
    @type mapx400: dns.name.Name object
    @see: RFC 2163"""
    __slots__ = ['preference', 'map822', 'mapx400']

    def __init__(self, rdclass, rdtype, preference, map822, mapx400):
        super(PX, self).__init__(rdclass, rdtype)
        self.preference = preference
        self.map822 = map822
        self.mapx400 = mapx400

    def to_text(self, origin=None, relativize=True, **kw):
        map822 = self.map822.choose_relativity(origin, relativize)
        mapx400 = self.mapx400.choose_relativity(origin, relativize)
        return '%d %s %s' % (self.preference, map822, mapx400)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True):
        preference = tok.get_uint16()
        map822 = tok.get_name()
        map822 = map822.choose_relativity(origin, relativize)
        mapx400 = tok.get_name(None)
        mapx400 = mapx400.choose_relativity(origin, relativize)
        tok.get_eol()
        return cls(rdclass, rdtype, preference, map822, mapx400)

    def to_wire(self, file, compress=None, origin=None):
        pref = struct.pack('!H', self.preference)
        file.write(pref)
        self.map822.to_wire(file, None, origin)
        self.mapx400.to_wire(file, None, origin)

    @classmethod
    def from_wire(cls, rdclass, rdtype, wire, current, rdlen, origin=None):
        preference, = struct.unpack('!H', wire[current:current + 2])
        current += 2
        rdlen -= 2
        map822, cused = dns.name.from_wire(wire[:current + rdlen], current)
        if cused > rdlen:
            raise dns.exception.FormError
        current += cused
        rdlen -= cused
        if origin is not None:
            map822 = map822.relativize(origin)
        mapx400, cused = dns.name.from_wire(wire[:current + rdlen], current)
        if cused != rdlen:
            raise dns.exception.FormError
        if origin is not None:
            mapx400 = mapx400.relativize(origin)
        return cls(rdclass, rdtype, preference, map822, mapx400)

    def choose_relativity(self, origin=None, relativize=True):
        self.map822 = self.map822.choose_relativity(origin, relativize)
        self.mapx400 = self.mapx400.choose_relativity(origin, relativize)