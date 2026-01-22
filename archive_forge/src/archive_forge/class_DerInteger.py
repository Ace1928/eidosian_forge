import struct
from Cryptodome.Util.py3compat import byte_string, bchr, bord
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
class DerInteger(DerObject):
    """Class to model a DER INTEGER.

        An example of encoding is::

          >>> from Cryptodome.Util.asn1 import DerInteger
          >>> from binascii import hexlify, unhexlify
          >>> int_der = DerInteger(9)
          >>> print hexlify(int_der.encode())

        which will show ``020109``, the DER encoding of 9.

        And for decoding::

          >>> s = unhexlify(b'020109')
          >>> try:
          >>>   int_der = DerInteger()
          >>>   int_der.decode(s)
          >>>   print int_der.value
          >>> except ValueError:
          >>>   print "Not a valid DER INTEGER"

        the output will be ``9``.

        :ivar value: The integer value
        :vartype value: integer
        """

    def __init__(self, value=0, implicit=None, explicit=None):
        """Initialize the DER object as an INTEGER.

                :Parameters:
                  value : integer
                    The value of the integer.

                  implicit : integer
                    The IMPLICIT tag to use for the encoded object.
                    It overrides the universal tag for INTEGER (2).
                """
        DerObject.__init__(self, 2, b'', implicit, False, explicit)
        self.value = value

    def encode(self):
        """Return the DER INTEGER, fully encoded as a
                binary string."""
        number = self.value
        self.payload = b''
        while True:
            self.payload = bchr(int(number & 255)) + self.payload
            if 128 <= number <= 255:
                self.payload = bchr(0) + self.payload
            if -128 <= number <= 255:
                break
            number >>= 8
        return DerObject.encode(self)

    def decode(self, der_encoded, strict=False):
        """Decode a DER-encoded INTEGER, and re-initializes this
                object with it.

                Args:
                  der_encoded (byte string): A complete INTEGER DER element.

                Raises:
                  ValueError: in case of parsing errors.
                """
        return DerObject.decode(self, der_encoded, strict=strict)

    def _decodeFromStream(self, s, strict):
        """Decode a complete DER INTEGER from a file."""
        DerObject._decodeFromStream(self, s, strict)
        if strict:
            if len(self.payload) == 0:
                raise ValueError('Invalid encoding for DER INTEGER: empty payload')
            if len(self.payload) >= 2 and struct.unpack('>H', self.payload[:2])[0] < 128:
                raise ValueError('Invalid encoding for DER INTEGER: leading zero')
        self.value = 0
        bits = 1
        for i in self.payload:
            self.value *= 256
            self.value += bord(i)
            bits <<= 8
        if self.payload and bord(self.payload[0]) & 128:
            self.value -= bits