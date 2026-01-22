from reportlab.graphics.shapes import *
from reportlab.graphics.renderbase import getStateDelta, renderScaledDrawing
from reportlab.pdfbase.pdfmetrics import getFont, unicode2T1
from reportlab.lib.utils import isUnicode
from reportlab import rl_config
from .utils import setFont as _setFont, RenderPMError
import os, sys
from io import BytesIO, StringIO
from math import sin, cos, pi, ceil
from reportlab.graphics.renderbase import Renderer
def _saveToBMP(self, f):
    """
        Niki Spahiev, <niki@vintech.bg>, asserts that this is a respectable way to get BMP without PIL
        f is a file like object to which the BMP is written
        """
    import struct
    gs = self._gs
    pix, width, height = (gs.pixBuf, gs.width, gs.height)
    f.write(struct.pack('=2sLLLLLLhh24x', 'BM', len(pix) + 54, 0, 54, 40, width, height, 1, 24))
    rowb = width * 3
    for o in range(len(pix), 0, -rowb):
        f.write(pix[o - rowb:o])
    f.write('\x00' * 14)