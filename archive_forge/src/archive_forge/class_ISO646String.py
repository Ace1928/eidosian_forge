import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import univ
class ISO646String(VisibleString):
    __doc__ = VisibleString.__doc__
    typeId = AbstractCharacterString.getTypeId()