from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import useful
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class TimestampTrustCondition(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.OptionalNamedType('ttsCertificateTrustTrees', CertificateTrustTrees().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.OptionalNamedType('ttsRevReq', CertRevReq().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))), namedtype.OptionalNamedType('ttsNameConstraints', NameConstraints().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2))), namedtype.OptionalNamedType('cautionPeriod', DeltaTime().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 3))), namedtype.OptionalNamedType('signatureTimestampDelay', DeltaTime().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 4))))