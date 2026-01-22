import string
from math import atan2
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .ScatterPlotItem import Symbols
from .TextItem import TextItem
from .UIGraphicsItem import UIGraphicsItem
from .ViewBox import ViewBox
class TargetLabel(TextItem):
    """A TextItem that attaches itself to a TargetItem.

    This class extends TextItem with the following features :
      * Automatically positions adjacent to the symbol at a fixed position.
      * Automatically reformats text when the symbol location has changed.

    Parameters
    ----------
    target : TargetItem
        The TargetItem to which this label will be attached to.
    text : str or callable, Optional
        Governs the text displayed, can be a fixed string or a format string
        that accepts the x, and y position of the target item; or be a callable
        method that accepts a tuple (x, y) and returns a string to be displayed.
        If None, an empty string is used.  Default is None
    offset : tuple or list or QPointF or QPoint
        Position to set the anchor of the TargetLabel away from the center of
        the target in pixels, by default it is (20, 0).
    anchor : tuple or list or QPointF or QPoint
        Position to rotate the TargetLabel about, and position to set the
        offset value to see :class:`~pyqtgraph.TextItem` for more information.
    kwargs : dict 
        kwargs contains arguments that are passed onto
        :class:`~pyqtgraph.TextItem` constructor, excluding text parameter
    """

    def __init__(self, target, text='', offset=(20, 0), anchor=(0, 0.5), **kwargs):
        if isinstance(offset, Point):
            self.offset = offset
        elif isinstance(offset, (tuple, list)):
            self.offset = Point(*offset)
        elif isinstance(offset, (QtCore.QPoint, QtCore.QPointF)):
            self.offset = Point(offset.x(), offset.y())
        else:
            raise TypeError('Offset parameter is the wrong data type')
        super().__init__(anchor=anchor, **kwargs)
        self.setParentItem(target)
        self.target = target
        self.setFormat(text)
        self.target.sigPositionChanged.connect(self.valueChanged)
        self.valueChanged()

    def format(self):
        return self._format

    def setFormat(self, text):
        """Method to set how the TargetLabel should display the text.  This
        method should be called from TargetItem.setLabel directly.

        Parameters
        ----------
        text : Callable or str
            Details how to format the text.
            If Callable, then the label will display the result of ``text(x, y)``
            If a fromatted string, then the output of ``text.format(x, y)`` will be
            displayed
            If a non-formatted string, then the text label will display ``text``
        """
        if not callable(text):
            parsed = list(string.Formatter().parse(text))
            if parsed and parsed[0][1] is not None:
                self.setProperty('formattableText', True)
            else:
                self.setText(text)
                self.setProperty('formattableText', False)
        else:
            self.setProperty('formattableText', False)
        self._format = text
        self.valueChanged()

    def valueChanged(self):
        x, y = self.target.pos()
        if self.property('formattableText'):
            self.setText(self._format.format(float(x), float(y)))
        elif callable(self._format):
            self.setText(self._format(x, y))

    def viewTransformChanged(self):
        viewbox = self.getViewBox()
        if isinstance(viewbox, ViewBox):
            viewPixelSize = viewbox.viewPixelSize()
            scaledOffset = QtCore.QPointF(self.offset.x() * viewPixelSize[0], self.offset.y() * viewPixelSize[1])
            self.setPos(scaledOffset)
        return super().viewTransformChanged()

    def mouseClickEvent(self, ev):
        return self.parentItem().mouseClickEvent(ev)

    def mouseDragEvent(self, ev):
        targetItem = self.parentItem()
        if not targetItem.movable or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        ev.accept()
        if ev.isStart():
            targetItem.symbolOffset = targetItem.pos() - self.mapToView(ev.buttonDownPos())
            targetItem.moving = True
        if not targetItem.moving:
            return
        targetItem.setPos(targetItem.symbolOffset + self.mapToView(ev.pos()))
        if ev.isFinish():
            targetItem.moving = False
            targetItem.sigPositionChangeFinished.emit(self)