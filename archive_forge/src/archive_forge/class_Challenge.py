from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc2314
from pyasn1_modules import rfc2459
from pyasn1_modules import rfc2511
class Challenge(univ.Sequence):
    """
    Challenge ::= SEQUENCE {
         owf                 AlgorithmIdentifier  OPTIONAL,
         witness             OCTET STRING,
         challenge           OCTET STRING
     }
    """
    componentType = namedtype.NamedTypes(namedtype.OptionalNamedType('owf', rfc2459.AlgorithmIdentifier()), namedtype.NamedType('witness', univ.OctetString()), namedtype.NamedType('challenge', univ.OctetString()))