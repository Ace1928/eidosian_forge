from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc2459
class BasicOCSPResponse(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('tbsResponseData', ResponseData()), namedtype.NamedType('signatureAlgorithm', rfc2459.AlgorithmIdentifier()), namedtype.NamedType('signature', univ.BitString()), namedtype.OptionalNamedType('certs', univ.SequenceOf(componentType=rfc2459.Certificate()).subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))))