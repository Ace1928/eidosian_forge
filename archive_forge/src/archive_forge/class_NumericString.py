import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import univ
class NumericString(AbstractCharacterString):
    __doc__ = AbstractCharacterString.__doc__
    tagSet = AbstractCharacterString.tagSet.tagImplicitly(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 18))
    encoding = 'us-ascii'
    typeId = AbstractCharacterString.getTypeId()