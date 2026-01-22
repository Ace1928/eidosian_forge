from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc2459
class OCSPResponse(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('responseStatus', OCSPResponseStatus()), namedtype.OptionalNamedType('responseBytes', ResponseBytes().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))))