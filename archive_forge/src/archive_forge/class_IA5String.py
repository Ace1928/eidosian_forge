import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import univ
class IA5String(AbstractCharacterString):
    __doc__ = AbstractCharacterString.__doc__
    tagSet = AbstractCharacterString.tagSet.tagImplicitly(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 22))
    encoding = 'us-ascii'
    typeId = AbstractCharacterString.getTypeId()