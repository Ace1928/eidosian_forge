from pyasn1_modules.rfc2459 import *
class DigestedData(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('version', Version()), namedtype.NamedType('digestAlgorithm', DigestAlgorithmIdentifier()), namedtype.NamedType('contentInfo', ContentInfo()), namedtype.NamedType('digest', Digest()))