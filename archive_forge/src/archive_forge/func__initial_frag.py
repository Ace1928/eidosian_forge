import re
import sys
import copy
import unicodedata
import reportlab.lib.sequencer
from reportlab.lib.abag import ABag
from reportlab.lib.utils import ImageReader, annotateException, encode_label, asUnicode
from reportlab.lib.colors import toColor, black
from reportlab.lib.fonts import tt2ps, ps2tt
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import inch,mm,cm,pica
from reportlab.rl_config import platypus_link_underline
from html.parser import HTMLParser
from html.entities import name2codepoint
def _initial_frag(self, attr, attrMap, bullet=0):
    style = self._style
    if attr != {}:
        style = copy.deepcopy(style)
        _applyAttributes(style, self.getAttributes(attr, attrMap))
        self._style = style
    frag = ParaFrag()
    frag.rise = 0
    frag.greek = 0
    frag.link = []
    try:
        if bullet:
            frag.fontName, frag.bold, frag.italic = ps2tt(style.bulletFontName)
            frag.fontSize = style.bulletFontSize
            frag.textColor = hasattr(style, 'bulletColor') and style.bulletColor or style.textColor
        else:
            frag.fontName, frag.bold, frag.italic = ps2tt(style.fontName)
            frag.fontSize = style.fontSize
            frag.textColor = style.textColor
    except:
        annotateException('error with style name=%s' % style.name)
    frag.us_lines = []
    self.nlinks = self.nlines = 0
    self._defaultLineWidths = dict(underline=getattr(style, 'underlineWidth', ''), strike=getattr(style, 'strikeWidth', ''))
    self._defaultLineColors = dict(underline=getattr(style, 'underlineColor', ''), strike=getattr(style, 'strikeColor', ''))
    self._defaultLineOffsets = dict(underline=getattr(style, 'underlineOffset', ''), strike=getattr(style, 'strikeOffset', ''))
    self._defaultLineGaps = dict(underline=getattr(style, 'underlineGap', ''), strike=getattr(style, 'strikeGap', ''))
    self._defaultLinkUnderline = getattr(style, 'linkUnderline', platypus_link_underline)
    return frag