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
def addArc(self, centerx, centery, radius, startangledegrees, endangledegrees, yradius=None, degreedelta=None, moveTo=None, reverse=None):
    P = getArcPoints(centerx, centery, radius, startangledegrees, endangledegrees, yradius=yradius, degreedelta=degreedelta, reverse=reverse)
    if moveTo or not len(self.operators):
        self.moveTo(P[0][0], P[0][1])
        del P[0]
    for x, y in P:
        self.lineTo(x, y)