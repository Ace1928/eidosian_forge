from pyasn1.type import univ, char, namedtype, namedval, tag, constraint, useful
from pyasn1_modules import rfc2560
from pyasn1_modules import rfc5280
class ServiceLocator(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('issuer', Name()), namedtype.NamedType('locator', AuthorityInfoAccessSyntax()))