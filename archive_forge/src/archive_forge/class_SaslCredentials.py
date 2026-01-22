from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
class SaslCredentials(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('mechanism', LDAPString()), namedtype.OptionalNamedType('credentials', univ.OctetString()))