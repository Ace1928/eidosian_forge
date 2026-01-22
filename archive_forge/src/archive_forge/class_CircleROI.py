import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
class CircleROI(EllipseROI):
    """
    Circular ROI subclass. Behaves exactly as EllipseROI, but may only be scaled
    proportionally to maintain its aspect ratio.
    
    ============== =============================================================
    **Arguments**
    pos            (length-2 sequence) The position of the ROI's origin.
    size           (length-2 sequence) The size of the ROI's bounding rectangle.
    \\**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    
    """

    def __init__(self, pos, size=None, radius=None, **args):
        if size is None:
            if radius is None:
                raise TypeError('Must provide either size or radius.')
            size = (radius * 2, radius * 2)
        EllipseROI.__init__(self, pos, size, aspectLocked=True, **args)

    def _addHandles(self):
        self.addScaleHandle([0.5 * 2.0 ** (-0.5) + 0.5, 0.5 * 2.0 ** (-0.5) + 0.5], [0.5, 0.5])