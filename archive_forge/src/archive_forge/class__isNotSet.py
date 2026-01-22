import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class _isNotSet(Validator):

    def test(self, x):
        return x is NotSetOr._not_set