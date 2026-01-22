import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class NumericAlign(str):
    """for creating the numeric string value for anchors etc etc
    dp is the character to align on (the last occurrence will be used)
    dpLen is the length of characters after the dp
    """

    def __new__(cls, dp='.', dpLen=0):
        self = str.__new__(cls, 'numeric')
        self._dp = dp
        self._dpLen = dpLen
        return self