from pyasn1.type import char
from pyasn1.type import namedval
from pyasn1.type import univ
from pyasn1_modules import rfc5755
class SecurityCategoryValues(univ.SequenceOf):
    componentType = char.UTF8String()