from pyasn1_modules import rfc2315
from pyasn1_modules.rfc2459 import *
class OptionalValidity(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.OptionalNamedType('notBefore', Time().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.OptionalNamedType('notAfter', Time().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))))