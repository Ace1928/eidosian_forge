import struct
from Cryptodome.Util.py3compat import byte_string, bchr, bord
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
def _decodeFromStream(self, s, strict):
    """Decode a complete DER SET OF from a file."""
    self._seq = []
    DerObject._decodeFromStream(self, s, strict)
    p = BytesIO_EOF(self.payload)
    setIdOctet = -1
    while p.remaining_data() > 0:
        p.set_bookmark()
        der = DerObject()
        der._decodeFromStream(p, strict)
        if setIdOctet < 0:
            setIdOctet = der._tag_octet
        elif setIdOctet != der._tag_octet:
            raise ValueError('Not all elements are of the same DER type')
        if setIdOctet != 2:
            self._seq.append(p.data_since_bookmark())
        else:
            derInt = DerInteger()
            derInt.decode(p.data_since_bookmark(), strict)
            self._seq.append(derInt.value)