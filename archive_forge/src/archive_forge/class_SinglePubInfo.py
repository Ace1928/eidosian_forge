from pyasn1_modules import rfc2315
from pyasn1_modules.rfc2459 import *
class SinglePubInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('pubMethod', univ.Integer(namedValues=namedval.NamedValues(('dontCare', 0), ('x500', 1), ('web', 2), ('ldap', 3)))), namedtype.OptionalNamedType('pubLocation', GeneralName()))