import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class matchesPattern(Validator):
    """Matches value, or its string representation, against regex"""

    def __init__(self, pattern):
        self._pattern = re.compile(pattern)

    def test(self, x):
        x = str(x)
        print('testing %s against %s' % (x, self._pattern))
        return self._pattern.match(x) != None