import os
from copy import deepcopy, copy
from reportlab.lib.colors import gray, lightgrey
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.styles import _baseFontName
from reportlab.lib.utils import strTypes, rl_safe_exec, annotateException
from reportlab.lib.abag import ABag
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ, overlapAttachedSpace, ignoreContainerActions, listWrapOnFakeWidth
from reportlab.lib.sequencer import _type2formatter
from reportlab.lib.styles import ListStyle
def _hmodel(s0, s1, h0, h1):
    a11 = 1.0 / s0 ** 2
    a12 = 1.0 / s0
    a21 = 1.0 / s1 ** 2
    a22 = 1.0 / s1
    det = a11 * a22 - a12 * a21
    b11 = a22 / det
    b12 = -a12 / det
    b21 = -a21 / det
    b22 = a11 / det
    a = b11 * h0 + b12 * h1
    b = b21 * h0 + b22 * h1
    return (a, b)