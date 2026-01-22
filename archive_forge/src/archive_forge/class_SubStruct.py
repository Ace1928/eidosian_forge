from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.ttLib import getSearchRange
from .otBase import (
from .otTables import (
from itertools import zip_longest
from functools import partial
import re
import struct
from typing import Optional
import logging
class SubStruct(Struct):

    def getConverter(self, tableType, lookupType):
        tableClass = self.lookupTypes[tableType][lookupType]
        return self.__class__(self.name, self.repeat, self.aux, tableClass)

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        super(SubStruct, self).xmlWrite(xmlWriter, font, value, None, attrs)