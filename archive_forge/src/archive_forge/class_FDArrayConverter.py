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
class FDArrayConverter(TableConverter):

    def _read(self, parent, value):
        try:
            vstore = parent.VarStore
        except AttributeError:
            vstore = None
        file = parent.file
        isCFF2 = parent._isCFF2
        file.seek(value)
        fdArray = FDArrayIndex(file, isCFF2=isCFF2)
        fdArray.vstore = vstore
        fdArray.strings = parent.strings
        fdArray.GlobalSubrs = parent.GlobalSubrs
        return fdArray

    def write(self, parent, value):
        return 0

    def xmlRead(self, name, attrs, content, parent):
        fdArray = FDArrayIndex()
        for element in content:
            if isinstance(element, str):
                continue
            name, attrs, content = element
            fdArray.fromXML(name, attrs, content)
        return fdArray