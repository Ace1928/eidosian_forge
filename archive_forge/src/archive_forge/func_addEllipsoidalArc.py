from reportlab.graphics.shapes import *
from reportlab.graphics.renderbase import getStateDelta, renderScaledDrawing
from reportlab.pdfbase.pdfmetrics import getFont, unicode2T1
from reportlab.lib.utils import isUnicode
from reportlab import rl_config
from .utils import setFont as _setFont, RenderPMError
import os, sys
from io import BytesIO, StringIO
from math import sin, cos, pi, ceil
from reportlab.graphics.renderbase import Renderer
def addEllipsoidalArc(self, cx, cy, rx, ry, ang1, ang2):
    """adds an ellisesoidal arc segment to a path, with an ellipse centered
        on cx,cy and with radii (major & minor axes) rx and ry.  The arc is
        drawn in the CCW direction.  Requires: (ang2-ang1) > 0"""
    (x0, y0), ctrlpts = self.bezierArcCCW(cx, cy, rx, ry, ang1, ang2)
    self.lineTo(x0, y0)
    for (x1, y1), (x2, y2), (x3, y3) in ctrlpts:
        self.curveTo(x1, y1, x2, y2, x3, y3)