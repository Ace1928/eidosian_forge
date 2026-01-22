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
class SubrsIndex(GlobalSubrsIndex):
    """This index contains a glyph's local subroutines. A local subroutine is a
    private set of ``CharString`` data which is accessible only to the glyph to
    which the index is attached."""
    compilerClass = SubrsCompiler