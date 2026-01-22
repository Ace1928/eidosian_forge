from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ
class OtherInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('keyInfo', KeySpecificInfo()), namedtype.OptionalNamedType('partyAInfo', univ.OctetString().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.NamedType('suppPubInfo', univ.OctetString().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))))