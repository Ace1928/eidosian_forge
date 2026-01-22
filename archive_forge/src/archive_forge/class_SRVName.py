from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class SRVName(char.IA5String):
    subtypeSpec = constraint.ValueSizeConstraint(1, MAX)