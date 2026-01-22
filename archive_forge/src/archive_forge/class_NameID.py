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
class NameID(UShort):

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.simpletag(name, attrs + [('value', value)])
        if font and value:
            nameTable = font.get('name')
            if nameTable:
                name = nameTable.getDebugName(value)
                xmlWriter.write('  ')
                if name:
                    xmlWriter.comment(name)
                else:
                    xmlWriter.comment('missing from name table')
                    log.warning('name id %d missing from name table' % value)
        xmlWriter.newline()