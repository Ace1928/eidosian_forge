from pyasn1_modules.rfc2459 import *
class IssuerAndSerialNumber(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('issuer', Name()), namedtype.NamedType('serialNumber', CertificateSerialNumber()))