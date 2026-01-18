from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
def wedgeDrawTimeCallback(self, node, canvas, renderer, **kwds):
    A = getattr(canvas, 'ctm', None)
    if not A:
        return
    if isinstance(node, Ellipse):
        c = self.pmcanv
        c.ellipse(node.cx, node.cy, node.rx, node.ry)
        p = c.vpath
        p = [(x[1], x[2]) for x in p]
    else:
        p = node.asPolygon().points
        p = [(p[i], p[i + 1]) for i in range(0, len(p), 2)]
    D = kwds.copy()
    D['poly'] = self.transformAndFlatten(A, p)
    return D