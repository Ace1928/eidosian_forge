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
class STATFlags(UShort):

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.simpletag(name, attrs + [('value', value)])
        flags = []
        if value & 1:
            flags.append('OlderSiblingFontAttribute')
        if value & 2:
            flags.append('ElidableAxisValueName')
        if flags:
            xmlWriter.write('  ')
            xmlWriter.comment(' '.join(flags))
        xmlWriter.newline()