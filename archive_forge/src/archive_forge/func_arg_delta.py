from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def arg_delta(self, name):
    valueList = self.popall()
    out = []
    if valueList and isinstance(valueList[0], list):
        out = valueList
    else:
        current = 0
        for v in valueList:
            current = current + v
            out.append(current)
    return out