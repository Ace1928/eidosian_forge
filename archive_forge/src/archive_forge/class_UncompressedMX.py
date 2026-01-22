from io import BytesIO
import struct
import dns.exception
import dns.rdata
import dns.name
class UncompressedMX(MXBase):
    """Base class for rdata that is like an MX record, but whose name
    is not compressed when converted to DNS wire format, and whose
    digestable form is not downcased."""

    def to_wire(self, file, compress=None, origin=None):
        super(UncompressedMX, self).to_wire(file, None, origin)

    def to_digestable(self, origin=None):
        f = BytesIO()
        self.to_wire(f, None, origin)
        return f.getvalue()