from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import useful
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class CommitmentType(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('identifier', CommitmentTypeIdentifier()), namedtype.OptionalNamedType('fieldOfApplication', FieldOfApplication().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.OptionalNamedType('semantics', DirectoryString().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))))