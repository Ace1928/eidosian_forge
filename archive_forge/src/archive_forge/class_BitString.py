import math
import sys
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import integer
from pyasn1.compat import octets
from pyasn1.type import base
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import tagmap
class BitString(base.SimpleAsn1Type):
    """Create |ASN.1| schema or value object.

    |ASN.1| class is based on :class:`~pyasn1.type.base.SimpleAsn1Type`, its
    objects are immutable and duck-type both Python :class:`tuple` (as a tuple
    of bits) and :class:`int` objects.

    Keyword Args
    ------------
    value: :class:`int`, :class:`str` or |ASN.1| object
        Python :class:`int` or :class:`str` literal representing binary
        or hexadecimal number or sequence of integer bits or |ASN.1| object.
        If `value` is not given, schema object will be created.

    tagSet: :py:class:`~pyasn1.type.tag.TagSet`
        Object representing non-default ASN.1 tag(s)

    subtypeSpec: :py:class:`~pyasn1.type.constraint.ConstraintsIntersection`
        Object representing non-default ASN.1 subtype constraint(s). Constraints
        verification for |ASN.1| type occurs automatically on object
        instantiation.

    namedValues: :py:class:`~pyasn1.type.namedval.NamedValues`
        Object representing non-default symbolic aliases for numbers

    binValue: :py:class:`str`
        Binary string initializer to use instead of the *value*.
        Example: '10110011'.

    hexValue: :py:class:`str`
        Hexadecimal string initializer to use instead of the *value*.
        Example: 'DEADBEEF'.

    Raises
    ------
    ~pyasn1.error.ValueConstraintError, ~pyasn1.error.PyAsn1Error
        On constraint violation or bad initializer.

    Examples
    --------
    .. code-block:: python

        class Rights(BitString):
            '''
            ASN.1 specification:

            Rights ::= BIT STRING { user-read(0), user-write(1),
                                    group-read(2), group-write(3),
                                    other-read(4), other-write(5) }

            group1 Rights ::= { group-read, group-write }
            group2 Rights ::= '0011'B
            group3 Rights ::= '3'H
            '''
            namedValues = NamedValues(
                ('user-read', 0), ('user-write', 1),
                ('group-read', 2), ('group-write', 3),
                ('other-read', 4), ('other-write', 5)
            )

        group1 = Rights(('group-read', 'group-write'))
        group2 = Rights('0011')
        group3 = Rights(0x3)
    """
    tagSet = tag.initTagSet(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 3))
    subtypeSpec = constraint.ConstraintsIntersection()
    namedValues = namedval.NamedValues()
    typeId = base.SimpleAsn1Type.getTypeId()
    defaultBinValue = defaultHexValue = noValue

    def __init__(self, value=noValue, **kwargs):
        if value is noValue:
            if kwargs:
                try:
                    value = self.fromBinaryString(kwargs.pop('binValue'), internalFormat=True)
                except KeyError:
                    pass
                try:
                    value = self.fromHexString(kwargs.pop('hexValue'), internalFormat=True)
                except KeyError:
                    pass
        if value is noValue:
            if self.defaultBinValue is not noValue:
                value = self.fromBinaryString(self.defaultBinValue, internalFormat=True)
            elif self.defaultHexValue is not noValue:
                value = self.fromHexString(self.defaultHexValue, internalFormat=True)
        if 'namedValues' not in kwargs:
            kwargs['namedValues'] = self.namedValues
        base.SimpleAsn1Type.__init__(self, value, **kwargs)

    def __str__(self):
        return self.asBinary()

    def __eq__(self, other):
        other = self.prettyIn(other)
        return self is other or (self._value == other and len(self._value) == len(other))

    def __ne__(self, other):
        other = self.prettyIn(other)
        return self._value != other or len(self._value) != len(other)

    def __lt__(self, other):
        other = self.prettyIn(other)
        return len(self._value) < len(other) or (len(self._value) == len(other) and self._value < other)

    def __le__(self, other):
        other = self.prettyIn(other)
        return len(self._value) <= len(other) or (len(self._value) == len(other) and self._value <= other)

    def __gt__(self, other):
        other = self.prettyIn(other)
        return len(self._value) > len(other) or (len(self._value) == len(other) and self._value > other)

    def __ge__(self, other):
        other = self.prettyIn(other)
        return len(self._value) >= len(other) or (len(self._value) == len(other) and self._value >= other)

    def __len__(self):
        return len(self._value)

    def __getitem__(self, i):
        if i.__class__ is slice:
            return self.clone([self[x] for x in range(*i.indices(len(self)))])
        else:
            length = len(self._value) - 1
            if i > length or i < 0:
                raise IndexError('bit index out of range')
            return self._value >> length - i & 1

    def __iter__(self):
        length = len(self._value)
        while length:
            length -= 1
            yield (self._value >> length & 1)

    def __reversed__(self):
        return reversed(tuple(self))

    def __add__(self, value):
        value = self.prettyIn(value)
        return self.clone(SizedInteger(self._value << len(value) | value).setBitLength(len(self._value) + len(value)))

    def __radd__(self, value):
        value = self.prettyIn(value)
        return self.clone(SizedInteger(value << len(self._value) | self._value).setBitLength(len(self._value) + len(value)))

    def __mul__(self, value):
        bitString = self._value
        while value > 1:
            bitString <<= len(self._value)
            bitString |= self._value
            value -= 1
        return self.clone(bitString)

    def __rmul__(self, value):
        return self * value

    def __lshift__(self, count):
        return self.clone(SizedInteger(self._value << count).setBitLength(len(self._value) + count))

    def __rshift__(self, count):
        return self.clone(SizedInteger(self._value >> count).setBitLength(max(0, len(self._value) - count)))

    def __int__(self):
        return int(self._value)

    def __float__(self):
        return float(self._value)
    if sys.version_info[0] < 3:

        def __long__(self):
            return self._value

    def asNumbers(self):
        """Get |ASN.1| value as a sequence of 8-bit integers.

        If |ASN.1| object length is not a multiple of 8, result
        will be left-padded with zeros.
        """
        return tuple(octets.octs2ints(self.asOctets()))

    def asOctets(self):
        """Get |ASN.1| value as a sequence of octets.

        If |ASN.1| object length is not a multiple of 8, result
        will be left-padded with zeros.
        """
        return integer.to_bytes(self._value, length=len(self))

    def asInteger(self):
        """Get |ASN.1| value as a single integer value.
        """
        return self._value

    def asBinary(self):
        """Get |ASN.1| value as a text string of bits.
        """
        binString = bin(self._value)[2:]
        return '0' * (len(self._value) - len(binString)) + binString

    @classmethod
    def fromHexString(cls, value, internalFormat=False, prepend=None):
        """Create a |ASN.1| object initialized from the hex string.

        Parameters
        ----------
        value: :class:`str`
            Text string like 'DEADBEEF'
        """
        try:
            value = SizedInteger(value, 16).setBitLength(len(value) * 4)
        except ValueError:
            raise error.PyAsn1Error('%s.fromHexString() error: %s' % (cls.__name__, sys.exc_info()[1]))
        if prepend is not None:
            value = SizedInteger(SizedInteger(prepend) << len(value) | value).setBitLength(len(prepend) + len(value))
        if not internalFormat:
            value = cls(value)
        return value

    @classmethod
    def fromBinaryString(cls, value, internalFormat=False, prepend=None):
        """Create a |ASN.1| object initialized from a string of '0' and '1'.

        Parameters
        ----------
        value: :class:`str`
            Text string like '1010111'
        """
        try:
            value = SizedInteger(value or '0', 2).setBitLength(len(value))
        except ValueError:
            raise error.PyAsn1Error('%s.fromBinaryString() error: %s' % (cls.__name__, sys.exc_info()[1]))
        if prepend is not None:
            value = SizedInteger(SizedInteger(prepend) << len(value) | value).setBitLength(len(prepend) + len(value))
        if not internalFormat:
            value = cls(value)
        return value

    @classmethod
    def fromOctetString(cls, value, internalFormat=False, prepend=None, padding=0):
        """Create a |ASN.1| object initialized from a string.

        Parameters
        ----------
        value: :class:`str` (Py2) or :class:`bytes` (Py3)
            Text string like '\\\\x01\\\\xff' (Py2) or b'\\\\x01\\\\xff' (Py3)
        """
        value = SizedInteger(integer.from_bytes(value) >> padding).setBitLength(len(value) * 8 - padding)
        if prepend is not None:
            value = SizedInteger(SizedInteger(prepend) << len(value) | value).setBitLength(len(prepend) + len(value))
        if not internalFormat:
            value = cls(value)
        return value

    def prettyIn(self, value):
        if isinstance(value, SizedInteger):
            return value
        elif octets.isStringType(value):
            if not value:
                return SizedInteger(0).setBitLength(0)
            elif value[0] == "'":
                if value[-2:] == "'B":
                    return self.fromBinaryString(value[1:-2], internalFormat=True)
                elif value[-2:] == "'H":
                    return self.fromHexString(value[1:-2], internalFormat=True)
                else:
                    raise error.PyAsn1Error('Bad BIT STRING value notation %s' % (value,))
            elif self.namedValues and (not value.isdigit()):
                names = [x.strip() for x in value.split(',')]
                try:
                    bitPositions = [self.namedValues[name] for name in names]
                except KeyError:
                    raise error.PyAsn1Error('unknown bit name(s) in %r' % (names,))
                rightmostPosition = max(bitPositions)
                number = 0
                for bitPosition in bitPositions:
                    number |= 1 << rightmostPosition - bitPosition
                return SizedInteger(number).setBitLength(rightmostPosition + 1)
            elif value.startswith('0x'):
                return self.fromHexString(value[2:], internalFormat=True)
            elif value.startswith('0b'):
                return self.fromBinaryString(value[2:], internalFormat=True)
            else:
                return self.fromBinaryString(value, internalFormat=True)
        elif isinstance(value, (tuple, list)):
            return self.fromBinaryString(''.join([b and '1' or '0' for b in value]), internalFormat=True)
        elif isinstance(value, BitString):
            return SizedInteger(value).setBitLength(len(value))
        elif isinstance(value, intTypes):
            return SizedInteger(value)
        else:
            raise error.PyAsn1Error("Bad BitString initializer type '%s'" % (value,))