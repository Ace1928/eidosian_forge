from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ
class IpAddress(univ.OctetString):
    tagSet = univ.OctetString.tagSet.tagImplicitly(tag.Tag(tag.tagClassApplication, tag.tagFormatSimple, 0))
    subtypeSpec = univ.Integer.subtypeSpec + constraint.ValueSizeConstraint(4, 4)