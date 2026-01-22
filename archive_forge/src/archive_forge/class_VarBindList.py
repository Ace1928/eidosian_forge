from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc1902
class VarBindList(univ.SequenceOf):
    componentType = VarBind()
    subtypeSpec = univ.SequenceOf.subtypeSpec + constraint.ValueSizeConstraint(0, max_bindings)