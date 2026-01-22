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
class _ExValidate:
    """class for syntax checking attributes
    """

    def __init__(self, tag, attr):
        self.tag = tag
        self.attr = attr

    def invalid(self, s):
        raise ValueError('<%s> invalid value %r for attribute %s' % (self.tag, s, self.attr))

    def validate(self, parser, s):
        raise ValueError('abstract method called')
        return s

    def __call__(self, parser, s):
        try:
            return self.validate(parser, s)
        except:
            self.invalid(s)