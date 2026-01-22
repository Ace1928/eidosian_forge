import struct
from Cryptodome.Util.py3compat import byte_string, bchr, bord
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
class DerOctetString(DerObject):
    """Class to model a DER OCTET STRING.

    An example of encoding is:

    >>> from Cryptodome.Util.asn1 import DerOctetString
    >>> from binascii import hexlify, unhexlify
    >>> os_der = DerOctetString(b'\\xaa')
    >>> os_der.payload += b'\\xbb'
    >>> print hexlify(os_der.encode())

    which will show ``0402aabb``, the DER encoding for the byte string
    ``b'\\xAA\\xBB'``.

    For decoding:

    >>> s = unhexlify(b'0402aabb')
    >>> try:
    >>>   os_der = DerOctetString()
    >>>   os_der.decode(s)
    >>>   print hexlify(os_der.payload)
    >>> except ValueError:
    >>>   print "Not a valid DER OCTET STRING"

    the output will be ``aabb``.

    :ivar payload: The content of the string
    :vartype payload: byte string
    """

    def __init__(self, value=b'', implicit=None):
        """Initialize the DER object as an OCTET STRING.

        :Parameters:
          value : byte string
            The initial payload of the object.
            If not specified, the payload is empty.

          implicit : integer
            The IMPLICIT tag to use for the encoded object.
            It overrides the universal tag for OCTET STRING (4).
        """
        DerObject.__init__(self, 4, value, implicit, False)