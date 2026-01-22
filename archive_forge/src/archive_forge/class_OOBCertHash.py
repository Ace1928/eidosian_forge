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
class OOBCertHash(univ.Sequence):
    """
    OOBCertHash ::= SEQUENCE {
         hashAlg     [0] AlgorithmIdentifier     OPTIONAL,
         certId      [1] CertId                  OPTIONAL,
         hashVal         BIT STRING
     }
    """
    componentType = namedtype.NamedTypes(namedtype.OptionalNamedType('hashAlg', rfc2459.AlgorithmIdentifier().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))), namedtype.OptionalNamedType('certId', rfc2511.CertId().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))), namedtype.NamedType('hashVal', univ.BitString()))