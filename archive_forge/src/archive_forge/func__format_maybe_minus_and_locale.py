import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
def _format_maybe_minus_and_locale(self, fmt, arg):
    """
        Format *arg* with *fmt*, applying Unicode minus and locale if desired.
        """
    return self.fix_minus((','.join((locale.format_string(part, (arg,), True).replace(',', '{,}') for part in fmt.split(','))) if self._useMathText else locale.format_string(fmt, (arg,), True)) if self._useLocale else fmt % arg)