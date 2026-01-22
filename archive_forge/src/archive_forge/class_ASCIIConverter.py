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
class ASCIIConverter(SimpleConverter):

    def _read(self, parent, value):
        return tostr(value, encoding='ascii')

    def write(self, parent, value):
        return tobytes(value, encoding='ascii')

    def xmlWrite(self, xmlWriter, name, value):
        xmlWriter.simpletag(name, value=tostr(value, encoding='ascii'))
        xmlWriter.newline()

    def xmlRead(self, name, attrs, content, parent):
        return tobytes(attrs['value'], encoding='ascii')