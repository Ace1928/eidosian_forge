from string import whitespace
from operator import truth
from unicodedata import category
from reportlab.pdfbase.pdfmetrics import stringWidth, getAscentDescent
from reportlab.platypus.paraparser import ParaParser, _PCT, _num as _parser_num, _re_us_value
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import Color
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.geomutils import normalizeTRBL
from reportlab.lib.textsplit import wordSplit, ALL_CANNOT_START
from reportlab.lib.styles import ParagraphStyle
from copy import deepcopy
from reportlab.lib.abag import ABag
from reportlab.rl_config import decimalSymbol, _FUZZ, paraFontSizeHeightOffset,\
from reportlab.lib.utils import _className, isBytes, isStr
from reportlab.lib.rl_accel import sameFrag
import re
from types import MethodType
class _SHYStr(str):
    """for simple soft hyphenated words"""

    def __new__(cls, s):
        S = s.split(_shy)
        if len(S) > 1:
            self = str.__new__(cls, u''.join(S))
            sp = [0]
            asp = sp.append
            for ss in S:
                asp(sp[-1] + len(ss))
            self.__sp__ = sp[1:-1]
        else:
            self = str.__new__(cls, s)
            self.__sp__ = []
        return self

    def __shysplit__(self, fontName, fontSize, baseWidth, limWidth, encoding='utf8'):
        """
            baseWidth = currentWidth + spaceWidth + hyphenWidth
            limWidth = maxWidth + spaceShrink
            """
        self._fsww = 2147483647
        for i, sp in reversed(list(enumerate(self.__sp__))):
            sw = self[:sp]
            sww = stringWidth(sw, fontName, fontSize, encoding)
            if not i:
                self._fsww = sww
            swnw = baseWidth + sww
            if swnw <= limWidth:
                T = self.__sp__[i:] + [len(self)]
                S = [self[T[j]:T[j + 1]] for j in range(len(T) - 1)]
                sw = _SHYStr(sw + u'-')
                sw.__sp__ = self.__sp__[:i]
                return [sw, _SHYStr(_shy.join(S))]