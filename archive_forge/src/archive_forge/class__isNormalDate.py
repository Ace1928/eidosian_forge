import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class _isNormalDate(Validator):

    def test(self, x):
        if isinstance(x, NormalDate):
            return True
        return x is not None and self.normalizeTest(x)

    def normalize(self, x):
        return NormalDate(x)