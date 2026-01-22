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
class SubrsConverter(TableConverter):

    def getClass(self):
        return SubrsIndex

    def _read(self, parent, value):
        file = parent.file
        isCFF2 = parent._isCFF2
        file.seek(parent.offset + value)
        return SubrsIndex(file, isCFF2=isCFF2)

    def write(self, parent, value):
        return 0