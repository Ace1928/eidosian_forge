from pyasn1.type import char
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class PermanentIdentifier(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.OptionalNamedType('identifierValue', char.UTF8String()), namedtype.OptionalNamedType('assigner', univ.ObjectIdentifier()))