from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1_modules import rfc1905
class SNMPv3Message(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('msgVersion', univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, 2147483647))), namedtype.NamedType('msgGlobalData', HeaderData()), namedtype.NamedType('msgSecurityParameters', univ.OctetString()), namedtype.NamedType('msgData', ScopedPduData()))