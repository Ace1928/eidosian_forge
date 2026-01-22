from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc8692
class KMACwithSHAKE256_params(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.DefaultedNamedType('kMACOutputLength', univ.Integer().subtype(value=512)), namedtype.DefaultedNamedType('customizationString', univ.OctetString().subtype(value='')))