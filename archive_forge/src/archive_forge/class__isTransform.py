import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class _isTransform(Validator):
    """Transform validator class."""

    def test(self, x):
        if isSeq(x):
            if len(x) == 6:
                for element in x:
                    if not isNumber(element):
                        return False
                return True
            else:
                return False
        else:
            return False