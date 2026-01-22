from pyasn1_modules import rfc2315
from pyasn1_modules.rfc2459 import *
class Controls(univ.SequenceOf):
    componentType = AttributeTypeAndValue()
    subtypeSpec = univ.SequenceOf.subtypeSpec + constraint.ValueSizeConstraint(1, MAX)