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
class GlobalSubrsIndex(Index):
    """This index contains all the global subroutines in the font. A global
    subroutine is a set of ``CharString`` data which is accessible to any
    glyph in the font, and are used to store repeated instructions - for
    example, components may be encoded as global subroutines, but so could
    hinting instructions.

    Remember that when interpreting a ``callgsubr`` instruction (or indeed
    a ``callsubr`` instruction) that you will need to add the "subroutine
    number bias" to number given:

    .. code:: python

            tt = ttLib.TTFont("Almendra-Bold.otf")
            u = tt["CFF "].cff[0].CharStrings["udieresis"]
            u.decompile()

            u.toXML(XMLWriter(sys.stdout))
            # <some stuff>
            # -64 callgsubr <-- Subroutine which implements the dieresis mark
            # <other stuff>

            tt["CFF "].cff[0].GlobalSubrs[-64] # <-- WRONG
            # <T2CharString (bytecode) at 103451d10>

            tt["CFF "].cff[0].GlobalSubrs[-64 + 107] # <-- RIGHT
            # <T2CharString (source) at 103451390>

    ("The bias applied depends on the number of subrs (gsubrs). If the number of
    subrs (gsubrs) is less than 1240, the bias is 107. Otherwise if it is less
    than 33900, it is 1131; otherwise it is 32768.",
    `Subroutine Operators <https://docs.microsoft.com/en-us/typography/opentype/otspec180/cff2charstr#section4.4>`)
    """
    compilerClass = GlobalSubrsCompiler
    subrClass = psCharStrings.T2CharString
    charStringClass = psCharStrings.T2CharString

    def __init__(self, file=None, globalSubrs=None, private=None, fdSelect=None, fdArray=None, isCFF2=None):
        super(GlobalSubrsIndex, self).__init__(file, isCFF2=isCFF2)
        self.globalSubrs = globalSubrs
        self.private = private
        if fdSelect:
            self.fdSelect = fdSelect
        if fdArray:
            self.fdArray = fdArray

    def produceItem(self, index, data, file, offset):
        if self.private is not None:
            private = self.private
        elif hasattr(self, 'fdArray') and self.fdArray is not None:
            if hasattr(self, 'fdSelect') and self.fdSelect is not None:
                fdIndex = self.fdSelect[index]
            else:
                fdIndex = 0
            private = self.fdArray[fdIndex].Private
        else:
            private = None
        return self.subrClass(data, private=private, globalSubrs=self.globalSubrs)

    def toXML(self, xmlWriter):
        """Write the subroutines index into XML representation onto the given
        :class:`fontTools.misc.xmlWriter.XMLWriter`.

        .. code:: python

                writer = xmlWriter.XMLWriter(sys.stdout)
                tt["CFF "].cff[0].GlobalSubrs.toXML(writer)

        """
        xmlWriter.comment("The 'index' attribute is only for humans; it is ignored when parsed.")
        xmlWriter.newline()
        for i in range(len(self)):
            subr = self[i]
            if subr.needsDecompilation():
                xmlWriter.begintag('CharString', index=i, raw=1)
            else:
                xmlWriter.begintag('CharString', index=i)
            xmlWriter.newline()
            subr.toXML(xmlWriter)
            xmlWriter.endtag('CharString')
            xmlWriter.newline()

    def fromXML(self, name, attrs, content):
        if name != 'CharString':
            return
        subr = self.subrClass()
        subr.fromXML(name, attrs, content)
        self.append(subr)

    def getItemAndSelector(self, index):
        sel = None
        if hasattr(self, 'fdSelect'):
            sel = self.fdSelect[index]
        return (self[index], sel)