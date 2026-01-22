import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import univ
class T61String(TeletexString):
    __doc__ = TeletexString.__doc__
    typeId = AbstractCharacterString.getTypeId()