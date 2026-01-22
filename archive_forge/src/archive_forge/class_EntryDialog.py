from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
class EntryDialog:
    """
    A dialog box for entering
    """

    def __init__(self, parent, original_text='', instructions='', set_callback=None, title=None):
        self._parent = parent
        self._original_text = original_text
        self._set_callback = set_callback
        width = int(max(30, len(original_text) * 3 / 2))
        self._top = Toplevel(parent)
        if title:
            self._top.title(title)
        entryframe = Frame(self._top)
        entryframe.pack(expand=1, fill='both', padx=5, pady=5, ipady=10)
        if instructions:
            l = Label(entryframe, text=instructions)
            l.pack(side='top', anchor='w', padx=30)
        self._entry = Entry(entryframe, width=width)
        self._entry.pack(expand=1, fill='x', padx=30)
        self._entry.insert(0, original_text)
        divider = Frame(self._top, borderwidth=1, relief='sunken')
        divider.pack(fill='x', ipady=1, padx=10)
        buttons = Frame(self._top)
        buttons.pack(expand=0, fill='x', padx=5, pady=5)
        b = Button(buttons, text='Cancel', command=self._cancel, width=8)
        b.pack(side='right', padx=5)
        b = Button(buttons, text='Ok', command=self._ok, width=8, default='active')
        b.pack(side='left', padx=5)
        b = Button(buttons, text='Apply', command=self._apply, width=8)
        b.pack(side='left')
        self._top.bind('<Return>', self._ok)
        self._top.bind('<Control-q>', self._cancel)
        self._top.bind('<Escape>', self._cancel)
        self._entry.focus()

    def _reset(self, *e):
        self._entry.delete(0, 'end')
        self._entry.insert(0, self._original_text)
        if self._set_callback:
            self._set_callback(self._original_text)

    def _cancel(self, *e):
        try:
            self._reset()
        except:
            pass
        self._destroy()

    def _ok(self, *e):
        self._apply()
        self._destroy()

    def _apply(self, *e):
        if self._set_callback:
            self._set_callback(self._entry.get())

    def _destroy(self, *e):
        if self._top is None:
            return
        self._top.destroy()
        self._top = None