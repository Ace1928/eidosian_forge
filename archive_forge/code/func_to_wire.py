import struct
import dns.exception
import dns.rdata
import dns.name
def to_wire(self, file, compress=None, origin=None):
    self.mname.to_wire(file, compress, origin)
    self.rname.to_wire(file, compress, origin)
    five_ints = struct.pack('!IIIII', self.serial, self.refresh, self.retry, self.expire, self.minimum)
    file.write(five_ints)