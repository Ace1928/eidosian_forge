from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
class SubstringFilter(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('type', AttributeDescription()), namedtype.NamedType('substrings', univ.SequenceOf(componentType=univ.Choice(componentType=namedtype.NamedTypes(namedtype.NamedType('initial', LDAPString().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.NamedType('any', LDAPString().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))), namedtype.NamedType('final', LDAPString().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))))))))