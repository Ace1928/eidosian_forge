import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class _isCodec(Validator):

    def test(self, x):
        if not isStr(x):
            return False
        try:
            a, b, c, d = codecs.lookup(x)
            return True
        except LookupError:
            return False