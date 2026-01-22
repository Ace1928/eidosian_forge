from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5751
from pyasn1_modules import rfc5480
from pyasn1_modules import rfc4055
from pyasn1_modules import rfc3279
class DSAKeyCapabilities(univ.Choice):
    componentType = namedtype.NamedTypes(namedtype.NamedType('keySizes', univ.Sequence(componentType=namedtype.NamedTypes(namedtype.NamedType('minKeySize', DSAKeySize()), namedtype.OptionalNamedType('maxKeySize', DSAKeySize()), namedtype.OptionalNamedType('maxSizeP', univ.Integer().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))), namedtype.OptionalNamedType('maxSizeQ', univ.Integer().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))), namedtype.OptionalNamedType('maxSizeG', univ.Integer().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3))))).subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))), namedtype.NamedType('keyParams', Dss_Parms().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))))