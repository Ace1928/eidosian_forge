import copy
import struct
from rdkit import DataStructs
def DetachVectsMatchingBit(self, bit):
    items = list(self.__vects.items())
    for k, v in items:
        if v.GetBit(bit):
            del self.__vects[k]
            self.__needReset = True