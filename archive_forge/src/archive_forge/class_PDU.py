from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc1902
class PDU(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('request-id', rfc1902.Integer32()), namedtype.NamedType('error-status', univ.Integer(namedValues=namedval.NamedValues(('noError', 0), ('tooBig', 1), ('noSuchName', 2), ('badValue', 3), ('readOnly', 4), ('genErr', 5), ('noAccess', 6), ('wrongType', 7), ('wrongLength', 8), ('wrongEncoding', 9), ('wrongValue', 10), ('noCreation', 11), ('inconsistentValue', 12), ('resourceUnavailable', 13), ('commitFailed', 14), ('undoFailed', 15), ('authorizationError', 16), ('notWritable', 17), ('inconsistentName', 18)))), namedtype.NamedType('error-index', univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, max_bindings))), namedtype.NamedType('variable-bindings', VarBindList()))