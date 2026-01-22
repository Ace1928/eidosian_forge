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
class _PCT(float):

    def __new__(cls, v):
        self = float.__new__(cls, v * 0.01)
        self._normalizer = 1.0
        self._value = v
        return self

    def normalizedValue(self, normalizer):
        if not normalizer:
            normaliser = self._normalizer
        r = _PCT(normalizer * self._value)
        r._value = self._value
        r._normalizer = normalizer
        return r

    def __copy__(self):
        r = _PCT(float(self))
        r._value = self._value
        r._normalizer = normalizer
        return r

    def __deepcopy__(self, mem):
        return self.__copy__()