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
def xmlToTag(tag):
    """The opposite of tagToXML()"""
    if tag == 'OS_2':
        return Tag('OS/2')
    if len(tag) == 8:
        return identifierToTag(tag)
    else:
        return Tag(tag + ' ' * (4 - len(tag)))