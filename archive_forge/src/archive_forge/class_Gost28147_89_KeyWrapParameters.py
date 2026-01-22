from pyasn1.type import univ, char, namedtype, namedval, tag, constraint, useful
from pyasn1_modules import rfc4357
from pyasn1_modules import rfc5280
class Gost28147_89_KeyWrapParameters(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('encryptionParamSet', Gost28147_89_ParamSet()), namedtype.OptionalNamedType('ukm', univ.OctetString().subtype(subtypeSpec=constraint.ValueSizeConstraint(8, 8))))