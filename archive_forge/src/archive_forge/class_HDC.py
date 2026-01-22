from __future__ import annotations
from . import Image
class HDC:
    """
    Wraps an HDC integer. The resulting object can be passed to the
    :py:meth:`~PIL.ImageWin.Dib.draw` and :py:meth:`~PIL.ImageWin.Dib.expose`
    methods.
    """

    def __init__(self, dc):
        self.dc = dc

    def __int__(self):
        return self.dc