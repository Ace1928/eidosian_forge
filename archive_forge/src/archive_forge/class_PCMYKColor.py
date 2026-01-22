import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
class PCMYKColor(CMYKColor):
    """100 based CMYKColor with density and a spotName; just like Rimas uses"""
    _scale = 100.0

    def __init__(self, cyan, magenta, yellow, black, density=100, spotName=None, knockout=None, alpha=100):
        CMYKColor.__init__(self, cyan / 100.0, magenta / 100.0, yellow / 100.0, black / 100.0, spotName, density / 100.0, knockout=knockout, alpha=alpha / 100.0)

    def __repr__(self):
        return '%s(%s%s%s%s%s)' % (self.__class__.__name__, fp_str(self.cyan * 100, self.magenta * 100, self.yellow * 100, self.black * 100).replace(' ', ','), self.spotName and ',spotName=' + repr(self.spotName) or '', self.density != 1 and ',density=' + fp_str(self.density * 100) or '', self.knockout is not None and ',knockout=%d' % self.knockout or '', self.alpha is not None and ',alpha=%s' % fp_str(self.alpha * 100) or '')

    def cKwds(self):
        K = self._cKwds
        S = K[:6]
        for k in self._cKwds:
            v = getattr(self, k)
            if k in S:
                v *= 100
            yield (k, v)
    cKwds = property(cKwds)