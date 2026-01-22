from pyasn1.type import univ, char, namedtype, namedval, tag, constraint, useful
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5480
from pyasn1_modules import rfc5652
from pyasn1_modules import rfc5751
from pyasn1_modules import rfc8018
class ECC_CMS_SharedInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('keyInfo', KeyWrapAlgorithm()), namedtype.OptionalNamedType('entityUInfo', univ.OctetString().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.NamedType('suppPubInfo', univ.OctetString().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))))