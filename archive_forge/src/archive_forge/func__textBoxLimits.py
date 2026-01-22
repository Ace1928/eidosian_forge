import os, sys
from math import pi, cos, sin, sqrt, radians, floor
from reportlab.platypus import Flowable
from reportlab.rl_config import shapeChecking, verbose, defaultGraphicsFontName as _baseGFontName, _unset_, decimalSymbol
from reportlab.lib import logger
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.utils import isSeq, asBytes
from reportlab.lib.attrmap import *
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.fonts import tt2ps
from reportlab.pdfgen.canvas import FILL_EVEN_ODD, FILL_NON_ZERO
from . transform import *
def _textBoxLimits(text, font, fontSize, leading, textAnchor, boxAnchor):
    w = 0
    for t in text:
        w = max(w, stringWidth(t, font, fontSize))
    h = len(text) * leading
    yt = fontSize
    if boxAnchor[0] == 's':
        yb = -h
        yt = yt - h
    elif boxAnchor[0] == 'n':
        yb = 0
    else:
        yb = -h / 2.0
        yt = yt + yb
    if boxAnchor[-1] == 'e':
        xb = -w
        if textAnchor == 'end':
            xt = 0
        elif textAnchor == 'start':
            xt = -w
        else:
            xt = -w / 2.0
    elif boxAnchor[-1] == 'w':
        xb = 0
        if textAnchor == 'end':
            xt = w
        elif textAnchor == 'start':
            xt = 0
        else:
            xt = w / 2.0
    else:
        xb = -w / 2.0
        if textAnchor == 'end':
            xt = -xb
        elif textAnchor == 'start':
            xt = xb
        else:
            xt = 0
    return (xb, yb, w, h, xt, yt)