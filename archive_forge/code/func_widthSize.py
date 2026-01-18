from __future__ import print_function
from reportlab.graphics.barcode.common import Barcode
from reportlab.lib.utils import asNative
def widthSize(self, value):
    self._sized = None
    self._widthSize = min(max(0, value), 1)