from __future__ import print_function
from reportlab.graphics.barcode.common import Barcode
from reportlab.lib.utils import asNative
def _getBarVInfo(self, y0=0):
    vInfo = {}
    hs = self.heightScale
    for b in ('T', 'D', 'A', 'F'):
        y = self.scale(b, self.bottoms, hs) + y0
        vInfo[b] = (y, self.scale(b, self.tops, hs) + y0 - y)
    return vInfo