import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
class TestROI(ROI):

    def __init__(self, pos, size, **args):
        ROI.__init__(self, pos, size, **args)
        self.addTranslateHandle([0.5, 0.5])
        self.addScaleHandle([1, 1], [0, 0])
        self.addScaleHandle([0, 0], [1, 1])
        self.addScaleRotateHandle([1, 0.5], [0.5, 0.5])
        self.addScaleHandle([0.5, 1], [0.5, 0.5])
        self.addRotateHandle([1, 0], [0, 0])
        self.addRotateHandle([0, 1], [1, 1])