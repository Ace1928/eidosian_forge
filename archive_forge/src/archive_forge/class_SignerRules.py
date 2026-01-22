from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import useful
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class SignerRules(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.OptionalNamedType('externalSignedData', univ.Boolean()), namedtype.NamedType('mandatedSignedAttr', CMSAttrs()), namedtype.NamedType('mandatedUnsignedAttr', CMSAttrs()), namedtype.DefaultedNamedType('mandatedCertificateRef', CertRefReq().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)).subtype(value='signerOnly')), namedtype.DefaultedNamedType('mandatedCertificateInfo', CertInfoReq().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)).subtype(value='none')), namedtype.OptionalNamedType('signPolExtensions', SignPolExtensions().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))))