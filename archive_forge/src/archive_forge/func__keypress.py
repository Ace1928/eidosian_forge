from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def _keypress(self, event):
    if event.keysym == 'Return' or event.keysym == 'space':
        insert_point = self._textwidget.index('insert')
        itemnum = int(insert_point.split('.')[0]) - 1
        self._fire_callback(event.keysym.lower(), itemnum)
        return
    elif event.keysym == 'Down':
        delta = '+1line'
    elif event.keysym == 'Up':
        delta = '-1line'
    elif event.keysym == 'Next':
        delta = '+10lines'
    elif event.keysym == 'Prior':
        delta = '-10lines'
    else:
        return 'continue'
    self._textwidget.mark_set('insert', 'insert' + delta)
    self._textwidget.see('insert')
    self._textwidget.tag_remove('sel', '1.0', 'end+1char')
    self._textwidget.tag_add('sel', 'insert linestart', 'insert lineend')
    insert_point = self._textwidget.index('insert')
    itemnum = int(insert_point.split('.')[0]) - 1
    self._fire_callback(event.keysym.lower(), itemnum)
    return 'break'