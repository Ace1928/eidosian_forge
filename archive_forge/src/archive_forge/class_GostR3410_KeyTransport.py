from pyasn1.type import univ, char, namedtype, namedval, tag, constraint, useful
from pyasn1_modules import rfc4357
from pyasn1_modules import rfc5280
class GostR3410_KeyTransport(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('sessionEncryptedKey', Gost28147_89_EncryptedKey()), namedtype.OptionalNamedType('transportParameters', GostR3410_TransportParameters().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))))