import struct
from Cryptodome.Util.py3compat import byte_string, bchr, bord
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
class DerBitString(DerObject):
    """Class to model a DER BIT STRING.

    An example of encoding is:

    >>> from Cryptodome.Util.asn1 import DerBitString
    >>> bs_der = DerBitString(b'\\xAA')
    >>> bs_der.value += b'\\xBB'
    >>> print(bs_der.encode().hex())

    which will show ``030300aabb``, the DER encoding for the bit string
    ``b'\\xAA\\xBB'``.

    For decoding:

    >>> s = bytes.fromhex('030300aabb')
    >>> try:
    >>>   bs_der = DerBitString()
    >>>   bs_der.decode(s)
    >>>   print(bs_der.value.hex())
    >>> except ValueError:
    >>>   print "Not a valid DER BIT STRING"

    the output will be ``aabb``.

    :ivar value: The content of the string
    :vartype value: byte string
    """

    def __init__(self, value=b'', implicit=None, explicit=None):
        """Initialize the DER object as a BIT STRING.

        :Parameters:
          value : byte string or DER object
            The initial, packed bit string.
            If not specified, the bit string is empty.
          implicit : integer
            The IMPLICIT tag to use for the encoded object.
            It overrides the universal tag for BIT STRING (3).
          explicit : integer
            The EXPLICIT tag to use for the encoded object.
        """
        DerObject.__init__(self, 3, b'', implicit, False, explicit)
        if isinstance(value, DerObject):
            self.value = value.encode()
        else:
            self.value = value

    def encode(self):
        """Return the DER BIT STRING, fully encoded as a
        byte string."""
        self.payload = b'\x00' + self.value
        return DerObject.encode(self)

    def decode(self, der_encoded, strict=False):
        """Decode a complete DER BIT STRING, and re-initializes this
        object with it.

        Args:
            der_encoded (byte string): a complete DER BIT STRING.
            strict (boolean):
                Whether decoding must check for strict DER compliancy.

        Raises:
            ValueError: in case of parsing errors.
        """
        return DerObject.decode(self, der_encoded, strict)

    def _decodeFromStream(self, s, strict):
        """Decode a complete DER BIT STRING DER from a file."""
        DerObject._decodeFromStream(self, s, strict)
        if self.payload and bord(self.payload[0]) != 0:
            raise ValueError('Not a valid BIT STRING')
        self.value = b''
        if self.payload:
            self.value = self.payload[1:]