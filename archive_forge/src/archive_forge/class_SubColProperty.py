import copy
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, OneOf, isString, isColorOrNone,\
from reportlab.lib.attrmap import *
from reportlab.pdfbase.pdfmetrics import stringWidth, getFont
from reportlab.graphics.widgetbase import Widget, TypedPropertyCollection, PropHolder
from reportlab.graphics.shapes import Drawing, Group, String, Rect, Line, STATE_DEFAULTS
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.utils import isSeq, find_locals, isStr, asNative
from reportlab.graphics.shapes import _baseGFontName
class SubColProperty(PropHolder):
    dividerLines = 0
    _attrMap = AttrMap(minWidth=AttrMapValue(isNumber, desc='minimum width for this subcol'), rpad=AttrMapValue(isNumber, desc='right padding for this subcol'), align=AttrMapValue(OneOf('left', 'right', 'center', 'centre', 'numeric'), desc='alignment in subCol'), fontName=AttrMapValue(isString, desc='Font name of the strings'), fontSize=AttrMapValue(isNumber, desc='Font size of the strings'), leading=AttrMapValue(isNumberOrNone, desc='leading for the strings'), fillColor=AttrMapValue(isColorOrNone, desc='fontColor'), underlines=AttrMapValue(EitherOr((NoneOr(isInstanceOf(Line)), SequenceOf(isInstanceOf(Line), emptyOK=0, lo=0, hi=2147483647))), desc='underline definitions'), overlines=AttrMapValue(EitherOr((NoneOr(isInstanceOf(Line)), SequenceOf(isInstanceOf(Line), emptyOK=0, lo=0, hi=2147483647))), desc='overline definitions'), dx=AttrMapValue(isNumber, desc='x offset from default position'), dy=AttrMapValue(isNumber, desc='y offset from default position'), vAlign=AttrMapValue(OneOf('top', 'bottom', 'middle'), desc='vertical alignment in the row'))