from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def _buttonpress(self, event):
    clickloc = '@%d,%d' % (event.x, event.y)
    insert_point = self._textwidget.index(clickloc)
    itemnum = int(insert_point.split('.')[0]) - 1
    self._fire_callback('click%d' % event.num, itemnum)