import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class OneOf(Validator):
    """Make validator functions for list of choices.

    Usage:
    f = reportlab.lib.validators.OneOf('happy','sad')
    or
    f = reportlab.lib.validators.OneOf(('happy','sad'))
    f('sad'),f('happy'), f('grumpy')
    (1,1,0)
    """

    def __init__(self, enum, *args):
        if isSeq(enum):
            if args != ():
                raise ValueError('Either all singleton args or a single sequence argument')
            self._enum = tuple(enum) + args
        else:
            self._enum = (enum,) + args
        self._patterns = tuple((_ for _ in self._enum if isinstance(_, _re_Pattern)))
        if self._patterns:
            self._enum = tuple((_ for _ in self._enum if not isinstance(_, _re_Pattern)))
            self.test = self._test_patterns

    def test(self, x):
        return x in self._enum

    def _test_patterns(self, x):
        v = x in self._enum
        if v:
            return True
        for p in self._patterns:
            v = p.match(x)
            if v:
                return True
        return False