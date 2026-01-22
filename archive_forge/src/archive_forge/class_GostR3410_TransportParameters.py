from pyasn1.type import univ, char, namedtype, namedval, tag, constraint, useful
from pyasn1_modules import rfc4357
from pyasn1_modules import rfc5280
class GostR3410_TransportParameters(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('encryptionParamSet', Gost28147_89_ParamSet()), namedtype.OptionalNamedType('ephemeralPublicKey', SubjectPublicKeyInfo().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.NamedType('ukm', univ.OctetString().subtype(subtypeSpec=constraint.ValueSizeConstraint(8, 8))))