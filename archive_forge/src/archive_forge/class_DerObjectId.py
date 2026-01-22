import struct
from Cryptodome.Util.py3compat import byte_string, bchr, bord
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
class DerObjectId(DerObject):
    """Class to model a DER OBJECT ID.

    An example of encoding is:

    >>> from Cryptodome.Util.asn1 import DerObjectId
    >>> from binascii import hexlify, unhexlify
    >>> oid_der = DerObjectId("1.2")
    >>> oid_der.value += ".840.113549.1.1.1"
    >>> print hexlify(oid_der.encode())

    which will show ``06092a864886f70d010101``, the DER encoding for the
    RSA Object Identifier ``1.2.840.113549.1.1.1``.

    For decoding:

    >>> s = unhexlify(b'06092a864886f70d010101')
    >>> try:
    >>>   oid_der = DerObjectId()
    >>>   oid_der.decode(s)
    >>>   print oid_der.value
    >>> except ValueError:
    >>>   print "Not a valid DER OBJECT ID"

    the output will be ``1.2.840.113549.1.1.1``.

    :ivar value: The Object ID (OID), a dot separated list of integers
    :vartype value: string
    """

    def __init__(self, value='', implicit=None, explicit=None):
        """Initialize the DER object as an OBJECT ID.

        :Parameters:
          value : string
            The initial Object Identifier (e.g. "1.2.0.0.6.2").
          implicit : integer
            The IMPLICIT tag to use for the encoded object.
            It overrides the universal tag for OBJECT ID (6).
          explicit : integer
            The EXPLICIT tag to use for the encoded object.
        """
        DerObject.__init__(self, 6, b'', implicit, False, explicit)
        self.value = value

    def encode(self):
        """Return the DER OBJECT ID, fully encoded as a
        binary string."""
        comps = [int(x) for x in self.value.split('.')]
        if len(comps) < 2:
            raise ValueError('Not a valid Object Identifier string')
        if comps[0] > 2:
            raise ValueError('First component must be 0, 1 or 2')
        if comps[0] < 2 and comps[1] > 39:
            raise ValueError('Second component must be 39 at most')
        subcomps = [40 * comps[0] + comps[1]] + comps[2:]
        encoding = []
        for v in reversed(subcomps):
            encoding.append(v & 127)
            v >>= 7
            while v:
                encoding.append(v & 127 | 128)
                v >>= 7
        self.payload = b''.join([bchr(x) for x in reversed(encoding)])
        return DerObject.encode(self)

    def decode(self, der_encoded, strict=False):
        """Decode a complete DER OBJECT ID, and re-initializes this
        object with it.

        Args:
            der_encoded (byte string):
                A complete DER OBJECT ID.
            strict (boolean):
                Whether decoding must check for strict DER compliancy.

        Raises:
            ValueError: in case of parsing errors.
        """
        return DerObject.decode(self, der_encoded, strict)

    def _decodeFromStream(self, s, strict):
        """Decode a complete DER OBJECT ID from a file."""
        DerObject._decodeFromStream(self, s, strict)
        p = BytesIO_EOF(self.payload)
        subcomps = []
        v = 0
        while p.remaining_data():
            c = p.read_byte()
            v = (v << 7) + (c & 127)
            if not c & 128:
                subcomps.append(v)
                v = 0
        if len(subcomps) == 0:
            raise ValueError('Empty payload')
        if subcomps[0] < 40:
            subcomps[:1] = [0, subcomps[0]]
        elif subcomps[0] < 80:
            subcomps[:1] = [1, subcomps[0] - 40]
        else:
            subcomps[:1] = [2, subcomps[0] - 80]
        self.value = '.'.join([str(x) for x in subcomps])