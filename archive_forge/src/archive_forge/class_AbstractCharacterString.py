import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import univ
class AbstractCharacterString(univ.OctetString):
    """Creates |ASN.1| schema or value object.

    |ASN.1| objects are immutable and duck-type Python 2 :class:`unicode` or Python 3 :class:`str`.
    When used in octet-stream context, |ASN.1| type assumes "|encoding|" encoding.

    Keyword Args
    ------------
    value: :class:`unicode`, :class:`str`, :class:`bytes` or |ASN.1| object
        unicode object (Python 2) or string (Python 3), alternatively string
        (Python 2) or bytes (Python 3) representing octet-stream of serialised
        unicode string (note `encoding` parameter) or |ASN.1| class instance.

    tagSet: :py:class:`~pyasn1.type.tag.TagSet`
        Object representing non-default ASN.1 tag(s)

    subtypeSpec: :py:class:`~pyasn1.type.constraint.ConstraintsIntersection`
        Object representing non-default ASN.1 subtype constraint(s)

    encoding: :py:class:`str`
        Unicode codec ID to encode/decode :class:`unicode` (Python 2) or
        :class:`str` (Python 3) the payload when |ASN.1| object is used
        in octet-stream context.

    Raises
    ------
    :py:class:`~pyasn1.error.PyAsn1Error`
        On constraint violation or bad initializer.
    """
    if sys.version_info[0] <= 2:

        def __str__(self):
            try:
                return self._value.encode(self.encoding)
            except UnicodeEncodeError:
                raise error.PyAsn1Error("Can't encode string '%s' with codec %s" % (self._value, self.encoding))

        def __unicode__(self):
            return unicode(self._value)

        def prettyIn(self, value):
            try:
                if isinstance(value, unicode):
                    return value
                elif isinstance(value, str):
                    return value.decode(self.encoding)
                elif isinstance(value, (tuple, list)):
                    return self.prettyIn(''.join([chr(x) for x in value]))
                elif isinstance(value, univ.OctetString):
                    return value.asOctets().decode(self.encoding)
                else:
                    return unicode(value)
            except (UnicodeDecodeError, LookupError):
                raise error.PyAsn1Error("Can't decode string '%s' with codec %s" % (value, self.encoding))

        def asOctets(self, padding=True):
            return str(self)

        def asNumbers(self, padding=True):
            return tuple([ord(x) for x in str(self)])
    else:

        def __str__(self):
            return str(self._value)

        def __bytes__(self):
            try:
                return self._value.encode(self.encoding)
            except UnicodeEncodeError:
                raise error.PyAsn1Error("Can't encode string '%s' with codec %s" % (self._value, self.encoding))

        def prettyIn(self, value):
            try:
                if isinstance(value, str):
                    return value
                elif isinstance(value, bytes):
                    return value.decode(self.encoding)
                elif isinstance(value, (tuple, list)):
                    return self.prettyIn(bytes(value))
                elif isinstance(value, univ.OctetString):
                    return value.asOctets().decode(self.encoding)
                else:
                    return str(value)
            except (UnicodeDecodeError, LookupError):
                raise error.PyAsn1Error("Can't decode string '%s' with codec %s" % (value, self.encoding))

        def asOctets(self, padding=True):
            return bytes(self)

        def asNumbers(self, padding=True):
            return tuple(bytes(self))

    def prettyOut(self, value):
        return value

    def prettyPrint(self, scope=0):
        value = self.prettyOut(self._value)
        if value is not self._value:
            return value
        return AbstractCharacterString.__str__(self)

    def __reversed__(self):
        return reversed(self._value)