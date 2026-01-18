from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
@staticmethod
def transformAndFlatten(A, p):
    """ transform an flatten a list of points
        A   transformation matrix
        p   points [(x0,y0),....(xk,yk).....]
        """
    if tuple(A) != (1, 0, 0, 1, 0, 0):
        iA = inverse(A)
        p = transformPoints(iA, p)
    return tuple(flatten(p))