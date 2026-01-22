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
class _CheckUS(_ExValidate):
    """class for syntax checking <u|strike> width/offset attributes"""

    def validate(self, parser, s):
        s = s.strip()
        if s:
            m = _re_us_value.match(s)
            if m:
                v = float(m.group(1))
                if m.group(2) == 'P':
                    return parser._stack[0].fontSize * v
            else:
                _num(s, allowRelative=False)
        return s