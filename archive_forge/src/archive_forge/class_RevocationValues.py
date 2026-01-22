from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import useful
from pyasn1.type import univ
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5652
from pyasn1_modules import rfc5035
from pyasn1_modules import rfc5755
from pyasn1_modules import rfc6960
from pyasn1_modules import rfc3161
class RevocationValues(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.OptionalNamedType('crlVals', univ.SequenceOf(componentType=CertificateList()).subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.OptionalNamedType('ocspVals', univ.SequenceOf(componentType=BasicOCSPResponse()).subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))), namedtype.OptionalNamedType('otherRevVals', OtherRevVals().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2))))