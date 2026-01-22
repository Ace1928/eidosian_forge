from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc5280
class BiometricData(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('typeOfBiometricData', TypeOfBiometricData()), namedtype.NamedType('hashAlgorithm', AlgorithmIdentifier()), namedtype.NamedType('biometricDataHash', univ.OctetString()), namedtype.OptionalNamedType('sourceDataUri', char.IA5String()))