from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def _find_room(self, widget, desired_x, desired_y):
    """
        Try to find a space for a given widget.
        """
    left, top, right, bot = self.scrollregion()
    w = widget.width()
    h = widget.height()
    if w >= right - left:
        return (0, 0)
    if h >= bot - top:
        return (0, 0)
    x1, y1, x2, y2 = widget.bbox()
    widget.move(left - x2 - 50, top - y2 - 50)
    if desired_x is not None:
        x = desired_x
        for y in range(top, bot - h, int((bot - top - h) / 10)):
            if not self._canvas.find_overlapping(x - 5, y - 5, x + w + 5, y + h + 5):
                return (x, y)
    if desired_y is not None:
        y = desired_y
        for x in range(left, right - w, int((right - left - w) / 10)):
            if not self._canvas.find_overlapping(x - 5, y - 5, x + w + 5, y + h + 5):
                return (x, y)
    for y in range(top, bot - h, int((bot - top - h) / 10)):
        for x in range(left, right - w, int((right - left - w) / 10)):
            if not self._canvas.find_overlapping(x - 5, y - 5, x + w + 5, y + h + 5):
                return (x, y)
    return (0, 0)