import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import univ
class BMPString(AbstractCharacterString):
    __doc__ = AbstractCharacterString.__doc__
    tagSet = AbstractCharacterString.tagSet.tagImplicitly(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 30))
    encoding = 'utf-16-be'
    typeId = AbstractCharacterString.getTypeId()