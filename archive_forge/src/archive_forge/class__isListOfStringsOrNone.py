import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class _isListOfStringsOrNone(Validator):
    """ListOfStringsOrNone validator class."""

    def test(self, x):
        if x is None:
            return True
        return isListOfStrings(x)