import os, sys
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.utils import asNative, base64_decodebytes
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.shapes import *
import unittest
from reportlab.rl_config import register_reset
from reportlab.graphics.widgets.signsandsymbols import SmileyFace
def _evalFuncDrawing(name, D, l=None, g=None):
    if g is None:
        g = globals()
    if l is None:
        l = locals()
    func = l.get(name, g.get(name, None))
    try:
        d = func()
    except:
        d = getFailedDrawing(name)
    D.append((d, getattr(func, '.__doc__', ''), name[3:]))