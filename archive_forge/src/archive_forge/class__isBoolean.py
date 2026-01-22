import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class _isBoolean(Validator):

    def test(self, x):
        if isinstance(int, bool):
            return x in (0, 1)
        return self.normalizeTest(x)

    def normalize(self, x):
        if x in (0, 1):
            return x
        try:
            S = x.upper()
        except:
            raise ValueError('Must be boolean not %s' % ascii(s))
        if S in ('YES', 'TRUE'):
            return True
        if S in ('NO', 'FALSE', None):
            return False
        raise ValueError('Must be boolean not %s' % ascii(s))