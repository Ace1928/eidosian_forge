import os
from copy import deepcopy, copy
from reportlab.lib.colors import gray, lightgrey
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.styles import _baseFontName
from reportlab.lib.utils import strTypes, rl_safe_exec, annotateException
from reportlab.lib.abag import ABag
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ, overlapAttachedSpace, ignoreContainerActions, listWrapOnFakeWidth
from reportlab.lib.sequencer import _type2formatter
from reportlab.lib.styles import ListStyle
class _ExtendBG(NullDraw):
    _ZEROSIZE = 1
    _SPACETRANSFER = True

    def __init__(self, y, height, bg, frame):
        self._y = y
        self._height = height
        self._bg = bg

    def wrap(self, availWidth, availHeight):
        return (0, 0)

    def frameAction(self, frame):
        bg = self._bg
        fby = self._y
        fbh = self._height
        fbgl = bg.left
        fbw = frame._width - fbgl - bg.right
        fbx = frame._x1 - fbgl
        canv = self.canv
        pn = canv.getPageNumber()
        bg.render(canv, frame, fbx, fby, fbw, fbh)