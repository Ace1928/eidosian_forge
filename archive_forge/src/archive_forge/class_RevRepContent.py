from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc2314
from pyasn1_modules import rfc2459
from pyasn1_modules import rfc2511
class RevRepContent(univ.Sequence):
    """
    RevRepContent ::= SEQUENCE {
         status       SEQUENCE SIZE (1..MAX) OF PKIStatusInfo,
         revCerts [0] SEQUENCE SIZE (1..MAX) OF CertId
                                             OPTIONAL,
         crls     [1] SEQUENCE SIZE (1..MAX) OF CertificateList
                                             OPTIONAL
    """
    componentType = namedtype.NamedTypes(namedtype.NamedType('status', PKIStatusInfo()), namedtype.OptionalNamedType('revCerts', univ.SequenceOf(componentType=rfc2511.CertId()).subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX), explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))), namedtype.OptionalNamedType('crls', univ.SequenceOf(componentType=rfc2459.CertificateList()).subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX), explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))))