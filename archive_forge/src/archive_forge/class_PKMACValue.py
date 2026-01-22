from pyasn1_modules import rfc2315
from pyasn1_modules.rfc2459 import *
class PKMACValue(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('algId', AlgorithmIdentifier()), namedtype.NamedType('value', univ.BitString()))