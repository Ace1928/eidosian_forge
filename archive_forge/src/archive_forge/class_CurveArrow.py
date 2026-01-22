import weakref
from math import atan2, degrees
from ..functions import clip_scalar
from ..Qt import QtCore, QtWidgets
from . import ArrowItem
from .GraphicsObject import GraphicsObject
class CurveArrow(CurvePoint):
    """Provides an arrow that points to any specific sample on a PlotCurveItem.
    Provides properties that can be animated."""

    def __init__(self, curve, index=0, pos=None, **opts):
        CurvePoint.__init__(self, curve, index=index, pos=pos)
        if opts.get('pxMode', True):
            opts['pxMode'] = False
            self.setFlags(self.flags() | self.GraphicsItemFlag.ItemIgnoresTransformations)
        opts['angle'] = 0
        self.arrow = ArrowItem.ArrowItem(**opts)
        self.arrow.setParentItem(self)

    def setStyle(self, **opts):
        return self.arrow.setStyle(**opts)