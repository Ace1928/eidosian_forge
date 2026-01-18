import struct
import dns.exception
import dns.rdata
import dns.name
def to_digestable(self, origin=None):
    return self.mname.to_digestable(origin) + self.rname.to_digestable(origin) + struct.pack('!IIIII', self.serial, self.refresh, self.retry, self.expire, self.minimum)