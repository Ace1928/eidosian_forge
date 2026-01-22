from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class AuthenticationContext(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('contextType', char.UTF8String()), namedtype.OptionalNamedType('contextInfo', char.UTF8String()))