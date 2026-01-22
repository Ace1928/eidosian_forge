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
class OctetString(base.SimpleAsn1Type):
    """Create |ASN.1| schema or value object.

    |ASN.1| class is based on :class:`~pyasn1.type.base.SimpleAsn1Type`, its
    objects are immutable and duck-type Python 2 :class:`str` or
    Python 3 :class:`bytes`. When used in Unicode context, |ASN.1| type
    assumes "|encoding|" serialisation.

    Keyword Args
    ------------
    value: :class:`unicode`, :class:`str`, :class:`bytes` or |ASN.1| object
        class:`str` (Python 2) or :class:`bytes` (Python 3), alternatively
        class:`unicode` object (Python 2) or :class:`str` (Python 3)
        representing character string to be serialised into octets
        (note `encoding` parameter) or |ASN.1| object.
        If `value` is not given, schema object will be created.

    tagSet: :py:class:`~pyasn1.type.tag.TagSet`
        Object representing non-default ASN.1 tag(s)

    subtypeSpec: :py:class:`~pyasn1.type.constraint.ConstraintsIntersection`
        Object representing non-default ASN.1 subtype constraint(s). Constraints
        verification for |ASN.1| type occurs automatically on object
        instantiation.

    encoding: :py:class:`str`
        Unicode codec ID to encode/decode :class:`unicode` (Python 2) or
        :class:`str` (Python 3) the payload when |ASN.1| object is used
        in text string context.

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

        class Icon(OctetString):
            '''
            ASN.1 specification:

            Icon ::= OCTET STRING

            icon1 Icon ::= '001100010011001000110011'B
            icon2 Icon ::= '313233'H
            '''
        icon1 = Icon.fromBinaryString('001100010011001000110011')
        icon2 = Icon.fromHexString('313233')
    """
    tagSet = tag.initTagSet(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 4))
    subtypeSpec = constraint.ConstraintsIntersection()
    typeId = base.SimpleAsn1Type.getTypeId()
    defaultBinValue = defaultHexValue = noValue
    encoding = 'iso-8859-1'

    def __init__(self, value=noValue, **kwargs):
        if kwargs:
            if value is noValue:
                try:
                    value = self.fromBinaryString(kwargs.pop('binValue'))
                except KeyError:
                    pass
                try:
                    value = self.fromHexString(kwargs.pop('hexValue'))
                except KeyError:
                    pass
        if value is noValue:
            if self.defaultBinValue is not noValue:
                value = self.fromBinaryString(self.defaultBinValue)
            elif self.defaultHexValue is not noValue:
                value = self.fromHexString(self.defaultHexValue)
        if 'encoding' not in kwargs:
            kwargs['encoding'] = self.encoding
        base.SimpleAsn1Type.__init__(self, value, **kwargs)
    if sys.version_info[0] <= 2:

        def prettyIn(self, value):
            if isinstance(value, str):
                return value
            elif isinstance(value, unicode):
                try:
                    return value.encode(self.encoding)
                except (LookupError, UnicodeEncodeError):
                    exc = sys.exc_info()[1]
                    raise error.PyAsn1UnicodeEncodeError("Can't encode string '%s' with codec %s" % (value, self.encoding), exc)
            elif isinstance(value, (tuple, list)):
                try:
                    return ''.join([chr(x) for x in value])
                except ValueError:
                    raise error.PyAsn1Error("Bad %s initializer '%s'" % (self.__class__.__name__, value))
            else:
                return str(value)

        def __str__(self):
            return str(self._value)

        def __unicode__(self):
            try:
                return self._value.decode(self.encoding)
            except UnicodeDecodeError:
                exc = sys.exc_info()[1]
                raise error.PyAsn1UnicodeDecodeError("Can't decode string '%s' with codec %s" % (self._value, self.encoding), exc)

        def asOctets(self):
            return str(self._value)

        def asNumbers(self):
            return tuple([ord(x) for x in self._value])
    else:

        def prettyIn(self, value):
            if isinstance(value, bytes):
                return value
            elif isinstance(value, str):
                try:
                    return value.encode(self.encoding)
                except UnicodeEncodeError:
                    exc = sys.exc_info()[1]
                    raise error.PyAsn1UnicodeEncodeError("Can't encode string '%s' with '%s' codec" % (value, self.encoding), exc)
            elif isinstance(value, OctetString):
                return value.asOctets()
            elif isinstance(value, base.SimpleAsn1Type):
                return self.prettyIn(str(value))
            elif isinstance(value, (tuple, list)):
                return self.prettyIn(bytes(value))
            else:
                return bytes(value)

        def __str__(self):
            try:
                return self._value.decode(self.encoding)
            except UnicodeDecodeError:
                exc = sys.exc_info()[1]
                raise error.PyAsn1UnicodeDecodeError("Can't decode string '%s' with '%s' codec at '%s'" % (self._value, self.encoding, self.__class__.__name__), exc)

        def __bytes__(self):
            return bytes(self._value)

        def asOctets(self):
            return bytes(self._value)

        def asNumbers(self):
            return tuple(self._value)

    def prettyOut(self, value):
        return value

    def prettyPrint(self, scope=0):
        value = self.prettyOut(self._value)
        if value is not self._value:
            return value
        numbers = self.asNumbers()
        for x in numbers:
            if x < 32 or x > 126:
                return '0x' + ''.join(('%.2x' % x for x in numbers))
        else:
            return OctetString.__str__(self)

    @staticmethod
    def fromBinaryString(value):
        """Create a |ASN.1| object initialized from a string of '0' and '1'.

        Parameters
        ----------
        value: :class:`str`
            Text string like '1010111'
        """
        bitNo = 8
        byte = 0
        r = []
        for v in value:
            if bitNo:
                bitNo -= 1
            else:
                bitNo = 7
                r.append(byte)
                byte = 0
            if v in ('0', '1'):
                v = int(v)
            else:
                raise error.PyAsn1Error('Non-binary OCTET STRING initializer %s' % (v,))
            byte |= v << bitNo
        r.append(byte)
        return octets.ints2octs(r)

    @staticmethod
    def fromHexString(value):
        """Create a |ASN.1| object initialized from the hex string.

        Parameters
        ----------
        value: :class:`str`
            Text string like 'DEADBEEF'
        """
        r = []
        p = []
        for v in value:
            if p:
                r.append(int(p + v, 16))
                p = None
            else:
                p = v
        if p:
            r.append(int(p + '0', 16))
        return octets.ints2octs(r)

    def __len__(self):
        return len(self._value)

    def __getitem__(self, i):
        if i.__class__ is slice:
            return self.clone(self._value[i])
        else:
            return self._value[i]

    def __iter__(self):
        return iter(self._value)

    def __contains__(self, value):
        return value in self._value

    def __add__(self, value):
        return self.clone(self._value + self.prettyIn(value))

    def __radd__(self, value):
        return self.clone(self.prettyIn(value) + self._value)

    def __mul__(self, value):
        return self.clone(self._value * value)

    def __rmul__(self, value):
        return self * value

    def __int__(self):
        return int(self._value)

    def __float__(self):
        return float(self._value)

    def __reversed__(self):
        return reversed(self._value)