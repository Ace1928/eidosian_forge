from pyasn1.type import univ, char, namedtype, namedval, tag, constraint, useful
from pyasn1_modules import rfc2560
from pyasn1_modules import rfc5280
class PreferredSignatureAlgorithm(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('sigIdentifier', AlgorithmIdentifier()), namedtype.OptionalNamedType('certIdentifier', AlgorithmIdentifier()))