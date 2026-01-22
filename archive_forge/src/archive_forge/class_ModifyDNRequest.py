from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
class ModifyDNRequest(univ.Sequence):
    tagSet = univ.Sequence.tagSet.tagImplicitly(tag.Tag(tag.tagClassApplication, tag.tagFormatConstructed, 12))
    componentType = namedtype.NamedTypes(namedtype.NamedType('entry', LDAPDN()), namedtype.NamedType('newrdn', RelativeLDAPDN()), namedtype.NamedType('deleteoldrdn', univ.Boolean()), namedtype.OptionalNamedType('newSuperior', LDAPDN().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))))