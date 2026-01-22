from pyasn1_modules.rfc2459 import *
class ExtendedCertificateOrCertificate(univ.Choice):
    componentType = namedtype.NamedTypes(namedtype.NamedType('certificate', Certificate()), namedtype.NamedType('extendedCertificate', ExtendedCertificate().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))))