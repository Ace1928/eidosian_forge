from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
def addSubsetObjects(self, doc, fontname, subset):
    """Generate a TrueType font subset and add it to the PDF document.
        Returns a PDFReference to the new FontDescriptor object."""
    fontFile = pdfdoc.PDFStream()
    fontFile.content = self.makeSubset(subset)
    fontFile.dictionary['Length1'] = len(fontFile.content)
    if doc.compression:
        fontFile.filters = [pdfdoc.PDFZCompress]
    fontFileRef = doc.Reference(fontFile, 'fontFile:%s(%s)' % (self.filename, fontname))
    flags = self.flags & ~FF_NONSYMBOLIC
    flags = flags | FF_SYMBOLIC
    fontDescriptor = pdfdoc.PDFDictionary({'Type': '/FontDescriptor', 'Ascent': self.ascent, 'CapHeight': self.capHeight, 'Descent': self.descent, 'Flags': flags, 'FontBBox': pdfdoc.PDFArray(self.bbox), 'FontName': pdfdoc.PDFName(fontname), 'ItalicAngle': self.italicAngle, 'StemV': self.stemV, 'FontFile2': fontFileRef, 'MissingWidth': self.defaultWidth})
    return doc.Reference(fontDescriptor, 'fontDescriptor:' + fontname)