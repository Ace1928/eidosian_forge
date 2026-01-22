from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ
class KeySpecificInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('algorithm', univ.ObjectIdentifier()), namedtype.NamedType('counter', univ.OctetString().subtype(subtypeSpec=constraint.ValueSizeConstraint(4, 4))))