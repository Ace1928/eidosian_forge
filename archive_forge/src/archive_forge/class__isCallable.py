import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class _isCallable(Validator):

    def test(self, x):
        return hasattr(x, '__call__')