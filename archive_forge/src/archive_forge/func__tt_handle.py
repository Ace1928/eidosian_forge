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
def _tt_handle(self, tt):
    """Iterate through a pre-parsed tuple tree (e.g. from pyrxp)"""
    tag = tt[0]
    try:
        start = getattr(self, 'start_' + tag)
        end = getattr(self, 'end_' + tag)
    except AttributeError:
        if not self.ignoreUnknownTags:
            raise ValueError('Invalid tag "%s"' % tag)
        start = self.start_unknown
        end = self.end_unknown
    start(tt[1] or {})
    C = tt[2]
    if C:
        M = self._tt_handlers
        for c in C:
            M[isinstance(c, (list, tuple))](c)
    end()