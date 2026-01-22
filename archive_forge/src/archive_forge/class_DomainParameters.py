from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
class DomainParameters(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('p', univ.Integer()), namedtype.NamedType('g', univ.Integer()), namedtype.NamedType('q', univ.Integer()), namedtype.NamedType('j', univ.Integer()), namedtype.OptionalNamedType('validationParms', ValidationParms()))