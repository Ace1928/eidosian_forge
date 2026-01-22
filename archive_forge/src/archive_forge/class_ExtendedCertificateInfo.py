from pyasn1_modules.rfc2459 import *
class ExtendedCertificateInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('version', Version()), namedtype.NamedType('certificate', Certificate()), namedtype.NamedType('attributes', Attributes()))