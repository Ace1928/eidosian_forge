import struct
from pyu2f import errors
def ToByteArray(self):
    """Serialize the command.

    Encodes the command as per the U2F specs, using the standard
    ISO 7816-4 extended encoding.  All Commands expect data, so
    Le is always present.

    Returns:
      Python bytearray of the encoded command.
    """
    lc = self.InternalEncodeLc()
    out = bytearray(4)
    out[0] = self.cla
    out[1] = self.ins
    out[2] = self.p1
    out[3] = self.p2
    if self.data:
        out.extend(lc)
        out.extend(self.data)
        out.extend([0, 0])
    else:
        out.extend([0, 0, 0])
    return out