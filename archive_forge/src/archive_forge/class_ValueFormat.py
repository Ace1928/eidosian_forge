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
class ValueFormat(IntValue):
    staticSize = 2

    def __init__(self, name, repeat, aux, tableClass=None, *, description=''):
        BaseConverter.__init__(self, name, repeat, aux, tableClass, description=description)
        self.which = 'ValueFormat' + ('2' if name[-1] == '2' else '1')

    def read(self, reader, font, tableDict):
        format = reader.readUShort()
        reader[self.which] = ValueRecordFactory(format)
        return format

    def write(self, writer, font, tableDict, format, repeatIndex=None):
        writer.writeUShort(format)
        writer[self.which] = ValueRecordFactory(format)