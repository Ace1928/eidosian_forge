from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
class BracketWidget(AbstractContainerWidget):
    """
    A canvas widget that places a pair of brackets around a child
    widget.

    Attributes:
      - ``color``: The color used to draw the brackets.
      - ``width``: The width of the brackets.
      - ``draggable``: whether the text can be dragged by the user.
    """

    def __init__(self, canvas, child, **attribs):
        """
        Create a new bracket widget.

        :type canvas: Tkinter.Canvas
        :param canvas: This canvas widget's canvas.
        :param child: The child widget.  ``child`` must not have a
            parent.
        :type child: CanvasWidget
        :param attribs: The new canvas widget's attributes.
        """
        self._child = child
        self._obrack = canvas.create_line(1, 1, 1, 1, 1, 1, 1, 1)
        self._cbrack = canvas.create_line(1, 1, 1, 1, 1, 1, 1, 1)
        AbstractContainerWidget.__init__(self, canvas, child, **attribs)

    def __setitem__(self, attr, value):
        if attr == 'color':
            self.canvas().itemconfig(self._obrack, fill=value)
            self.canvas().itemconfig(self._cbrack, fill=value)
        elif attr == 'width':
            self.canvas().itemconfig(self._obrack, width=value)
            self.canvas().itemconfig(self._cbrack, width=value)
        else:
            CanvasWidget.__setitem__(self, attr, value)

    def __getitem__(self, attr):
        if attr == 'color':
            return self.canvas().itemcget(self._obrack, 'outline')
        elif attr == 'width':
            return self.canvas().itemcget(self._obrack, 'width')
        else:
            return CanvasWidget.__getitem__(self, attr)

    def _update(self, child):
        x1, y1, x2, y2 = child.bbox()
        width = max((y2 - y1) / 8, 2)
        self.canvas().coords(self._obrack, x1, y1, x1 - width, y1, x1 - width, y2, x1, y2)
        self.canvas().coords(self._cbrack, x2, y1, x2 + width, y1, x2 + width, y2, x2, y2)

    def _tags(self):
        return [self._obrack, self._cbrack]