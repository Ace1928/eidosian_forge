from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ
class ObjectSyntax(univ.Choice):
    componentType = namedtype.NamedTypes(namedtype.NamedType('simple', SimpleSyntax()), namedtype.NamedType('application-wide', ApplicationSyntax()))