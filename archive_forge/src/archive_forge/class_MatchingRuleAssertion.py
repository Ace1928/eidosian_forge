from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
class MatchingRuleAssertion(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.OptionalNamedType('matchingRule', MatchingRuleId().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))), namedtype.OptionalNamedType('type', AttributeDescription().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))), namedtype.NamedType('matchValue', AssertionValue().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3))), namedtype.DefaultedNamedType('dnAttributes', univ.Boolean('False').subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 4))))