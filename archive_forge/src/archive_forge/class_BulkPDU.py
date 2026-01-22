from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc1902
class BulkPDU(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('request-id', rfc1902.Integer32()), namedtype.NamedType('non-repeaters', univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, max_bindings))), namedtype.NamedType('max-repetitions', univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, max_bindings))), namedtype.NamedType('variable-bindings', VarBindList()))