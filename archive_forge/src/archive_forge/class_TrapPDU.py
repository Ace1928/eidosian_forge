from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc1155
class TrapPDU(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('enterprise', univ.ObjectIdentifier()), namedtype.NamedType('agent-addr', rfc1155.NetworkAddress()), namedtype.NamedType('generic-trap', univ.Integer().clone(namedValues=namedval.NamedValues(('coldStart', 0), ('warmStart', 1), ('linkDown', 2), ('linkUp', 3), ('authenticationFailure', 4), ('egpNeighborLoss', 5), ('enterpriseSpecific', 6)))), namedtype.NamedType('specific-trap', univ.Integer()), namedtype.NamedType('time-stamp', rfc1155.TimeTicks()), namedtype.NamedType('variable-bindings', VarBindList()))