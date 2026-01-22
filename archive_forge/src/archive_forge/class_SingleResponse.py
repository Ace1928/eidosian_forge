from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc2459
class SingleResponse(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('certID', CertID()), namedtype.NamedType('certStatus', CertStatus()), namedtype.NamedType('thisUpdate', useful.GeneralizedTime()), namedtype.OptionalNamedType('nextUpdate', useful.GeneralizedTime().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.OptionalNamedType('singleExtensions', rfc2459.Extensions().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))))