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
class Hatching(Path):
    """define a hatching of a set of polygons defined by lists of the form [x0,y0,x1,y1,....,xn,yn]"""
    _attrMap = AttrMap(BASE=Path, xyLists=AttrMapValue(EitherOr((isListOfNumbers, SequenceOf(isListOfNumbers, lo=1)), 'xy list(s)'), desc='list(s) of numbers in the form x1, y1, x2, y2 ... xn, yn'), angles=AttrMapValue(EitherOr((isNumber, isListOfNumbers, 'angle(s)')), desc='the angle or list of angles at which hatching lines should be drawn'), spacings=AttrMapValue(EitherOr((isNumber, isListOfNumbers, 'spacings(s)')), desc='orthogonal distance(s) between hatching lines'))

    def __init__(self, spacings=2, angles=45, xyLists=[], **kwds):
        Path.__init__(self, **kwds)
        if isListOfNumbers(xyLists):
            xyLists = (xyLists,)
        if isNumber(angles):
            angles = (angles,)
        if isNumber(spacings):
            spacings = (spacings,)
        i = len(angles) - len(spacings)
        if i > 0:
            spacings = list(spacings) + i * [spacings[-1]]
        self.xyLists = xyLists
        self.angles = angles
        self.spacings = spacings
        moveTo = self.moveTo
        lineTo = self.lineTo
        for i, theta in enumerate(angles):
            spacing = spacings[i]
            theta = radians(theta)
            cosTheta = cos(theta)
            sinTheta = sin(theta)
            spanMin = 2147483647
            spanMax = -spanMin
            for P in xyLists:
                for j in range(0, len(P), 2):
                    y = P[j + 1] * cosTheta - P[j] * sinTheta
                    spanMin = min(y, spanMin)
                    spanMax = max(y, spanMax)
            spanStart = int(floor(spanMin / spacing)) - 1
            spanEnd = int(floor(spanMax / spacing)) + 1
            for step in range(spanStart, spanEnd):
                nodeX = []
                stripeY = spacing * step
                for P in xyLists:
                    k = len(P) - 2
                    for j in range(0, len(P), 2):
                        a = P[k]
                        b = P[k + 1]
                        a1 = a * cosTheta + b * sinTheta
                        b1 = b * cosTheta - a * sinTheta
                        x = P[j]
                        y = P[j + 1]
                        x1 = x * cosTheta + y * sinTheta
                        y1 = y * cosTheta - x * sinTheta
                        if b1 < stripeY and y1 >= stripeY or (y1 < stripeY and b1 >= stripeY):
                            nodeX.append(a1 + (x1 - a1) * (stripeY - b1) / (y1 - b1))
                        k = j
                nodeX.sort()
                for j in range(0, len(nodeX), 2):
                    a = nodeX[j] * cosTheta - stripeY * sinTheta
                    b = stripeY * cosTheta + nodeX[j] * sinTheta
                    x = nodeX[j + 1] * cosTheta - stripeY * sinTheta
                    y = stripeY * cosTheta + nodeX[j + 1] * sinTheta
                    moveTo(a, b)
                    lineTo(x, y)