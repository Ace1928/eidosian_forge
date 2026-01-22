from pyasn1.type import char
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class SIM(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('hashAlg', rfc5280.AlgorithmIdentifier()), namedtype.NamedType('authorityRandom', univ.OctetString()), namedtype.NamedType('pEPSI', univ.OctetString()))