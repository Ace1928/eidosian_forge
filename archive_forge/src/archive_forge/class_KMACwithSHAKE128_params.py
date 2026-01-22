from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc8692
class KMACwithSHAKE128_params(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.DefaultedNamedType('kMACOutputLength', univ.Integer().subtype(value=256)), namedtype.DefaultedNamedType('customizationString', univ.OctetString().subtype(value='')))