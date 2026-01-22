from pyasn1_modules import rfc2315
from pyasn1_modules.rfc2459 import *
class POPOSigningKeyInput(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('authInfo', univ.Choice(componentType=namedtype.NamedTypes(namedtype.NamedType('sender', GeneralName().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.NamedType('publicKeyMAC', PKMACValue())))), namedtype.NamedType('publicKey', SubjectPublicKeyInfo()))