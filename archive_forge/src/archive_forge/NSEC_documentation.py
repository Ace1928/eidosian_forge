import struct
import dns.exception
import dns.rdata
import dns.rdatatype
import dns.name
from dns._compat import xrange
NSEC record

    @ivar next: the next name
    @type next: dns.name.Name object
    @ivar windows: the windowed bitmap list
    @type windows: list of (window number, string) tuples