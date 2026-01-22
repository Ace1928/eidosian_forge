import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
def _non_decade_format(self, sign_string, base, fx, usetex):
    """Return string for non-decade locations."""
    b = float(base)
    exponent = math.floor(fx)
    coeff = b ** (fx - exponent)
    if _is_close_to_int(coeff):
        coeff = round(coeff)
    return '$\\mathdefault{%s%g\\times%s^{%d}}$' % (sign_string, coeff, base, exponent)