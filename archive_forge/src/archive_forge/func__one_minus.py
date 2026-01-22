import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
def _one_minus(self, s):
    if self._use_overline:
        return '\\overline{%s}' % s
    else:
        return f'1-{s}'