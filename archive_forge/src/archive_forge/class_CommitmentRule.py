from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import useful
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class CommitmentRule(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('selCommitmentTypes', SelectedCommitmentTypes()), namedtype.OptionalNamedType('signerAndVeriferRules', SignerAndVerifierRules().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))), namedtype.OptionalNamedType('signingCertTrustCondition', SigningCertTrustCondition().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))), namedtype.OptionalNamedType('timeStampTrustCondition', TimestampTrustCondition().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2))), namedtype.OptionalNamedType('attributeTrustCondition', AttributeTrustCondition().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 3))), namedtype.OptionalNamedType('algorithmConstraintSet', AlgorithmConstraintSet().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 4))), namedtype.OptionalNamedType('signPolExtensions', SignPolExtensions().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 5))))