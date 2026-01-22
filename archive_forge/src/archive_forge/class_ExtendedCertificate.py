from pyasn1_modules.rfc2459 import *
class ExtendedCertificate(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('extendedCertificateInfo', ExtendedCertificateInfo()), namedtype.NamedType('signatureAlgorithm', SignatureAlgorithmIdentifier()), namedtype.NamedType('signature', Signature()))