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
class Any(OctetString):
    """Create |ASN.1| schema or value object.

    |ASN.1| class is based on :class:`~pyasn1.type.base.SimpleAsn1Type`,
    its objects are immutable and duck-type Python 2 :class:`str` or Python 3
    :class:`bytes`. When used in Unicode context, |ASN.1| type assumes
    "|encoding|" serialisation.

    Keyword Args
    ------------
    value: :class:`unicode`, :class:`str`, :class:`bytes` or |ASN.1| object
        :class:`str` (Python 2) or :class:`bytes` (Python 3), alternatively
        :class:`unicode` object (Python 2) or :class:`str` (Python 3)
        representing character string to be serialised into octets (note
        `encoding` parameter) or |ASN.1| object.
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

        class Error(Sequence):
            '''
            ASN.1 specification:

            Error ::= SEQUENCE {
                code      INTEGER,
                parameter ANY DEFINED BY code  -- Either INTEGER or REAL
            }
            '''
            componentType=NamedTypes(
                NamedType('code', Integer()),
                NamedType('parameter', Any(),
                          openType=OpenType('code', {1: Integer(),
                                                     2: Real()}))
            )

        error = Error()
        error['code'] = 1
        error['parameter'] = Integer(1234)
    """
    tagSet = tag.TagSet()
    subtypeSpec = constraint.ConstraintsIntersection()
    typeId = OctetString.getTypeId()

    @property
    def tagMap(self):
        """"Return a :class:`~pyasn1.type.tagmap.TagMap` object mapping
            ASN.1 tags to ASN.1 objects contained within callee.
        """
        try:
            return self._tagMap
        except AttributeError:
            self._tagMap = tagmap.TagMap({self.tagSet: self}, {eoo.endOfOctets.tagSet: eoo.endOfOctets}, self)
            return self._tagMap