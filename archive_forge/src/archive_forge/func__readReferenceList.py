from io import BytesIO
import struct
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytesjoin, tostr
from collections import OrderedDict
from collections.abc import MutableMapping
def _readReferenceList(self, resType, refListOffset, numRes):
    resources = []
    for i in range(numRes):
        refOffset = refListOffset + ResourceRefItemSize * i
        refData = self._read(ResourceRefItemSize, refOffset)
        res = Resource(resType)
        res.decompile(refData, self)
        resources.append(res)
    return resources