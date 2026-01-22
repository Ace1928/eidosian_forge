from pyasn1_modules.rfc2459 import *
class DigestInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('digestAlgorithm', DigestAlgorithmIdentifier()), namedtype.NamedType('digest', Digest()))