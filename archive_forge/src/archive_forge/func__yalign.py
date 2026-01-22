from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def _yalign(self, top, bot):
    if self._align == 'top':
        return top
    if self._align == 'bottom':
        return bot
    if self._align == 'center':
        return (top + bot) / 2