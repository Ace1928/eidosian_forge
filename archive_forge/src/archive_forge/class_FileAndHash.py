from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import useful
from pyasn1.type import univ
from pyasn1_modules import rfc5652
class FileAndHash(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('file', char.IA5String()), namedtype.NamedType('hash', univ.BitString()))