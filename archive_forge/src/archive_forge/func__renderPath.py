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
def _renderPath(path, drawFuncs, countOnly=False, forceClose=False):
    """Helper function for renderers."""
    points = path.points
    i = 0
    hadClosePath = 0
    hadMoveTo = 0
    active = not countOnly
    for op in path.operators:
        if op == _MOVETO:
            if forceClose:
                if hadMoveTo and pop != _CLOSEPATH:
                    hadClosePath += 1
                    if active:
                        drawFuncs[_CLOSEPATH]()
            hadMoveTo += 1
        nArgs = _PATH_OP_ARG_COUNT[op]
        j = i + nArgs
        drawFuncs[op](*points[i:j])
        i = j
        if op == _CLOSEPATH:
            hadClosePath += 1
        pop = op
    if forceClose and hadMoveTo and (pop != _CLOSEPATH):
        hadClosePath += 1
        if active:
            drawFuncs[_CLOSEPATH]()
    return hadMoveTo == hadClosePath