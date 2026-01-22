import itertools
from reportlab.platypus.flowables import Flowable
from reportlab.graphics.shapes import Group, Rect
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isNumberOrNone, isColor, Validator
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.units import mm
from reportlab.lib.utils import asUnicodeEx, isUnicode
from reportlab.graphics.barcode import qrencoder
class isUnicodeOrQRList(Validator):

    def _test(self, x):
        if isUnicode(x):
            return True
        if all((isinstance(v, qrencoder.QR) for v in x)):
            return True
        return False

    def test(self, x):
        return self._test(x) or self.normalizeTest(x)

    def normalize(self, x):
        if self._test(x):
            return x
        try:
            return asUnicodeEx(x)
        except UnicodeError:
            raise ValueError("Can't convert to unicode: %r" % x)