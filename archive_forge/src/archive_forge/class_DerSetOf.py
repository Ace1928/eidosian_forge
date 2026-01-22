import struct
from Cryptodome.Util.py3compat import byte_string, bchr, bord
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
class DerSetOf(DerObject):
    """Class to model a DER SET OF.

    An example of encoding is:

    >>> from Cryptodome.Util.asn1 import DerBitString
    >>> from binascii import hexlify, unhexlify
    >>> so_der = DerSetOf([4,5])
    >>> so_der.add(6)
    >>> print hexlify(so_der.encode())

    which will show ``3109020104020105020106``, the DER encoding
    of a SET OF with items 4,5, and 6.

    For decoding:

    >>> s = unhexlify(b'3109020104020105020106')
    >>> try:
    >>>   so_der = DerSetOf()
    >>>   so_der.decode(s)
    >>>   print [x for x in so_der]
    >>> except ValueError:
    >>>   print "Not a valid DER SET OF"

    the output will be ``[4, 5, 6]``.
    """

    def __init__(self, startSet=None, implicit=None):
        """Initialize the DER object as a SET OF.

        :Parameters:
          startSet : container
            The initial set of integers or DER encoded objects.
          implicit : integer
            The IMPLICIT tag to use for the encoded object.
            It overrides the universal tag for SET OF (17).
        """
        DerObject.__init__(self, 17, b'', implicit, True)
        self._seq = []
        self._elemOctet = None
        if startSet:
            for e in startSet:
                self.add(e)

    def __getitem__(self, n):
        return self._seq[n]

    def __iter__(self):
        return iter(self._seq)

    def __len__(self):
        return len(self._seq)

    def add(self, elem):
        """Add an element to the set.

        Args:
            elem (byte string or integer):
              An element of the same type of objects already in the set.
              It can be an integer or a DER encoded object.
        """
        if _is_number(elem):
            eo = 2
        elif isinstance(elem, DerObject):
            eo = self._tag_octet
        else:
            eo = bord(elem[0])
        if self._elemOctet != eo:
            if self._elemOctet is not None:
                raise ValueError('New element does not belong to the set')
            self._elemOctet = eo
        if elem not in self._seq:
            self._seq.append(elem)

    def decode(self, der_encoded, strict=False):
        """Decode a complete SET OF DER element, and re-initializes this
        object with it.

        DER INTEGERs are decoded into Python integers. Any other DER
        element is left undecoded; its validity is not checked.

        Args:
            der_encoded (byte string): a complete DER BIT SET OF.
            strict (boolean):
                Whether decoding must check for strict DER compliancy.

        Raises:
            ValueError: in case of parsing errors.
        """
        return DerObject.decode(self, der_encoded, strict)

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

    def encode(self):
        """Return this SET OF DER element, fully encoded as a
        binary string.
        """
        ordered = []
        for item in self._seq:
            if _is_number(item):
                bys = DerInteger(item).encode()
            elif isinstance(item, DerObject):
                bys = item.encode()
            else:
                bys = item
            ordered.append(bys)
        ordered.sort()
        self.payload = b''.join(ordered)
        return DerObject.encode(self)