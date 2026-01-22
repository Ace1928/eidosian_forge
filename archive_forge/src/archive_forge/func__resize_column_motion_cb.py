import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def _resize_column_motion_cb(self, event):
    lb = self._listboxes[self._resize_column_index]
    charwidth = lb.winfo_width() / lb['width']
    x1 = event.x + event.widget.winfo_x()
    x2 = lb.winfo_x() + lb.winfo_width()
    lb['width'] = max(3, lb['width'] + (x1 - x2) // charwidth)