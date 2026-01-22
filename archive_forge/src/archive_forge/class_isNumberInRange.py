import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class isNumberInRange(_isNumber):

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def test(self, x):
        try:
            n = self.normalize(x)
            if self.min <= n <= self.max:
                return True
        except ValueError:
            pass
        return False