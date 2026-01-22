import struct
import dns.exception
import dns.rdata
import dns.name
SOA record

    @ivar mname: the SOA MNAME (master name) field
    @type mname: dns.name.Name object
    @ivar rname: the SOA RNAME (responsible name) field
    @type rname: dns.name.Name object
    @ivar serial: The zone's serial number
    @type serial: int
    @ivar refresh: The zone's refresh value (in seconds)
    @type refresh: int
    @ivar retry: The zone's retry value (in seconds)
    @type retry: int
    @ivar expire: The zone's expiration value (in seconds)
    @type expire: int
    @ivar minimum: The zone's negative caching time (in seconds, called
    "minimum" for historical reasons)
    @type minimum: int
    @see: RFC 1035