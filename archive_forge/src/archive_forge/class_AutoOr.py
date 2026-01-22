import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class AutoOr(EitherOr):

    def test(self, x):
        return isAuto(x) or super().test(x)