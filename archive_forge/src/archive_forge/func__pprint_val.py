import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
def _pprint_val(self, x, d):
    if abs(x) < 10000.0 and x == int(x):
        return '%d' % x
    fmt = '%1.3e' if d < 0.01 else '%1.3f' if d <= 1 else '%1.2f' if d <= 10 else '%1.1f' if d <= 100000.0 else '%1.1e'
    s = fmt % x
    tup = s.split('e')
    if len(tup) == 2:
        mantissa = tup[0].rstrip('0').rstrip('.')
        exponent = int(tup[1])
        if exponent:
            s = '%se%d' % (mantissa, exponent)
        else:
            s = mantissa
    else:
        s = s.rstrip('0').rstrip('.')
    return s