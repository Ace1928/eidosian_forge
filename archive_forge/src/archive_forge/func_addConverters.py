from fontTools.misc import sstruct
from fontTools.misc import psCharStrings
from fontTools.misc.arrayTools import unionRect, intRect
from fontTools.misc.textTools import (
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.ttLib.tables.otBase import OTTableReader
from fontTools.ttLib.tables import otTables as ot
from io import BytesIO
import struct
import logging
import re
def addConverters(table):
    for i in range(len(table)):
        op, name, arg, default, conv = table[i]
        if conv is not None:
            continue
        if arg in ('delta', 'array'):
            conv = ArrayConverter()
        elif arg == 'number':
            conv = NumberConverter()
        elif arg == 'SID':
            conv = ASCIIConverter()
        elif arg == 'blendList':
            conv = None
        else:
            assert False
        table[i] = (op, name, arg, default, conv)