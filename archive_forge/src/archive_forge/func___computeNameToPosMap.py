import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import tagmap
def __computeNameToPosMap(self):
    nameToPosMap = {}
    for idx, namedType in enumerate(self.__namedTypes):
        if namedType.name in nameToPosMap:
            return NamedTypes.PostponedError('Duplicate component name %s at %s' % (namedType.name, namedType))
        nameToPosMap[namedType.name] = idx
    return nameToPosMap