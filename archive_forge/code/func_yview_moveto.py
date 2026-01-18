import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def yview_moveto(self, *args, **kwargs):
    for lb in self._listboxes:
        lb.yview_moveto(*args, **kwargs)