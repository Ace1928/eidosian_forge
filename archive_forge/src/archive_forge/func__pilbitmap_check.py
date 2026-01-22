from __future__ import annotations
import tkinter
from io import BytesIO
from . import Image
def _pilbitmap_check():
    global _pilbitmap_ok
    if _pilbitmap_ok is None:
        try:
            im = Image.new('1', (1, 1))
            tkinter.BitmapImage(data=f'PIL:{im.im.id}')
            _pilbitmap_ok = 1
        except tkinter.TclError:
            _pilbitmap_ok = 0
    return _pilbitmap_ok