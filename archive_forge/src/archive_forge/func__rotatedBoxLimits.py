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
def _rotatedBoxLimits(x, y, w, h, angle):
    """
    Find the corner points of the rotated w x h sized box at x,y
    return the corner points and the min max points in the original space
    """
    C = zTransformPoints(rotate(angle), ((x, y), (x + w, y), (x + w, y + h), (x, y + h)))
    X = [x[0] for x in C]
    Y = [x[1] for x in C]
    return (min(X), max(X), min(Y), max(Y), C)