from __future__ import annotations
import tkinter
from io import BytesIO
from . import Image
def _pyimagingtkcall(command, photo, id):
    tk = photo.tk
    try:
        tk.call(command, photo, id)
    except tkinter.TclError:
        from . import _imagingtk
        _imagingtk.tkinit(tk.interpaddr())
        tk.call(command, photo, id)