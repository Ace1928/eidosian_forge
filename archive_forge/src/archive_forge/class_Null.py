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
class Null(OctetString):
    """Create |ASN.1| schema or value object.

    |ASN.1| class is based on :class:`~pyasn1.type.base.SimpleAsn1Type`, its
    objects are immutable and duck-type Python :class:`str` objects
    (always empty).

    Keyword Args
    ------------
    value: :class:`str` or |ASN.1| object
        Python empty :class:`str` literal or any object that evaluates to :obj:`False`
        If `value` is not given, schema object will be created.

    tagSet: :py:class:`~pyasn1.type.tag.TagSet`
        Object representing non-default ASN.1 tag(s)

    Raises
    ------
    ~pyasn1.error.ValueConstraintError, ~pyasn1.error.PyAsn1Error
        On constraint violation or bad initializer.

    Examples
    --------
    .. code-block:: python

        class Ack(Null):
            '''
            ASN.1 specification:

            Ack ::= NULL
            '''
        ack = Ack('')
    """
    tagSet = tag.initTagSet(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 5))
    subtypeSpec = OctetString.subtypeSpec + constraint.SingleValueConstraint(octets.str2octs(''))
    typeId = OctetString.getTypeId()

    def prettyIn(self, value):
        if value:
            return value
        return octets.str2octs('')