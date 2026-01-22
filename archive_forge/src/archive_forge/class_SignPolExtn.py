from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import useful
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class SignPolExtn(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('extnID', univ.ObjectIdentifier()), namedtype.NamedType('extnValue', univ.OctetString()))