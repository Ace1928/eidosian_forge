from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class ProxyCertInfoExtension(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.OptionalNamedType('pCPathLenConstraint', ProxyCertPathLengthConstraint()), namedtype.NamedType('proxyPolicy', ProxyPolicy()))