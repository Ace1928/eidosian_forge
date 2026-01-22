from fontTools.config import Config
from fontTools.misc import xmlWriter
from fontTools.misc.configTools import AbstractConfig
from fontTools.misc.textTools import Tag, byteord, tostr
from fontTools.misc.loggingTools import deprecateArgument
from fontTools.ttLib import TTLibError
from fontTools.ttLib.ttGlyphSet import _TTGlyph, _TTGlyphSetCFF, _TTGlyphSetGlyf
from fontTools.ttLib.sfnt import SFNTReader, SFNTWriter
from io import BytesIO, StringIO, UnsupportedOperation
import os
import logging
import traceback
def _tableToXML(self, writer, tag, quiet=None, splitGlyphs=False):
    if quiet is not None:
        deprecateArgument('quiet', 'configure logging instead')
    if tag in self:
        table = self[tag]
        report = "Dumping '%s' table..." % tag
    else:
        report = "No '%s' table found." % tag
    log.info(report)
    if tag not in self:
        return
    xmlTag = tagToXML(tag)
    attrs = dict()
    if hasattr(table, 'ERROR'):
        attrs['ERROR'] = 'decompilation error'
    from .tables.DefaultTable import DefaultTable
    if table.__class__ == DefaultTable:
        attrs['raw'] = True
    writer.begintag(xmlTag, **attrs)
    writer.newline()
    if tag == 'glyf':
        table.toXML(writer, self, splitGlyphs=splitGlyphs)
    else:
        table.toXML(writer, self)
    writer.endtag(xmlTag)
    writer.newline()
    writer.newline()