from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def _init_itemframe(self, options):
    self._itemframe = Frame(self._parent)
    options.setdefault('background', '#e0e0e0')
    self._textwidget = Text(self._itemframe, **options)
    self._textscroll = Scrollbar(self._itemframe, takefocus=0, orient='vertical')
    self._textwidget.config(yscrollcommand=self._textscroll.set)
    self._textscroll.config(command=self._textwidget.yview)
    self._textscroll.pack(side='right', fill='y')
    self._textwidget.pack(expand=1, fill='both', side='left')
    self._textwidget.tag_config('highlight', background='#e0ffff', border='1', relief='raised')
    self._init_colortags(self._textwidget, options)
    self._textwidget.tag_config('sel', foreground='')
    self._textwidget.tag_config('sel', foreground='', background='', border='', underline=1)
    self._textwidget.tag_lower('highlight', 'sel')