from pyasn1.type import univ, char, namedtype, namedval, tag, constraint, useful
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5480
from pyasn1_modules import rfc5652
from pyasn1_modules import rfc5751
from pyasn1_modules import rfc8018
class MQVuserKeyingMaterial(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('ephemeralPublicKey', OriginatorPublicKey()), namedtype.OptionalNamedType('addedukm', UserKeyingMaterial().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))))