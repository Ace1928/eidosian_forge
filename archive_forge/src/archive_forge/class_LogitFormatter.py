import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
class LogitFormatter(Formatter):
    """
    Probability formatter (using Math text).
    """

    def __init__(self, *, use_overline=False, one_half='\\frac{1}{2}', minor=False, minor_threshold=25, minor_number=6):
        """
        Parameters
        ----------
        use_overline : bool, default: False
            If x > 1/2, with x = 1-v, indicate if x should be displayed as
            $\\overline{v}$. The default is to display $1-v$.

        one_half : str, default: r"\\frac{1}{2}"
            The string used to represent 1/2.

        minor : bool, default: False
            Indicate if the formatter is formatting minor ticks or not.
            Basically minor ticks are not labelled, except when only few ticks
            are provided, ticks with most space with neighbor ticks are
            labelled. See other parameters to change the default behavior.

        minor_threshold : int, default: 25
            Maximum number of locs for labelling some minor ticks. This
            parameter have no effect if minor is False.

        minor_number : int, default: 6
            Number of ticks which are labelled when the number of ticks is
            below the threshold.
        """
        self._use_overline = use_overline
        self._one_half = one_half
        self._minor = minor
        self._labelled = set()
        self._minor_threshold = minor_threshold
        self._minor_number = minor_number

    def use_overline(self, use_overline):
        """
        Switch display mode with overline for labelling p>1/2.

        Parameters
        ----------
        use_overline : bool, default: False
            If x > 1/2, with x = 1-v, indicate if x should be displayed as
            $\\overline{v}$. The default is to display $1-v$.
        """
        self._use_overline = use_overline

    def set_one_half(self, one_half):
        """
        Set the way one half is displayed.

        one_half : str, default: r"\\frac{1}{2}"
            The string used to represent 1/2.
        """
        self._one_half = one_half

    def set_minor_threshold(self, minor_threshold):
        """
        Set the threshold for labelling minors ticks.

        Parameters
        ----------
        minor_threshold : int
            Maximum number of locations for labelling some minor ticks. This
            parameter have no effect if minor is False.
        """
        self._minor_threshold = minor_threshold

    def set_minor_number(self, minor_number):
        """
        Set the number of minor ticks to label when some minor ticks are
        labelled.

        Parameters
        ----------
        minor_number : int
            Number of ticks which are labelled when the number of ticks is
            below the threshold.
        """
        self._minor_number = minor_number

    def set_locs(self, locs):
        self.locs = np.array(locs)
        self._labelled.clear()
        if not self._minor:
            return None
        if all((_is_decade(x, rtol=1e-07) or _is_decade(1 - x, rtol=1e-07) or (_is_close_to_int(2 * x) and int(np.round(2 * x)) == 1) for x in locs)):
            return None
        if len(locs) < self._minor_threshold:
            if len(locs) < self._minor_number:
                self._labelled.update(locs)
            else:
                diff = np.diff(-np.log(1 / self.locs - 1))
                space_pessimistic = np.minimum(np.concatenate(((np.inf,), diff)), np.concatenate((diff, (np.inf,))))
                space_sum = np.concatenate(((0,), diff)) + np.concatenate((diff, (0,)))
                good_minor = sorted(range(len(self.locs)), key=lambda i: (space_pessimistic[i], space_sum[i]))[-self._minor_number:]
                self._labelled.update((locs[i] for i in good_minor))

    def _format_value(self, x, locs, sci_notation=True):
        if sci_notation:
            exponent = math.floor(np.log10(x))
            min_precision = 0
        else:
            exponent = 0
            min_precision = 1
        value = x * 10 ** (-exponent)
        if len(locs) < 2:
            precision = min_precision
        else:
            diff = np.sort(np.abs(locs - x))[1]
            precision = -np.log10(diff) + exponent
            precision = int(np.round(precision)) if _is_close_to_int(precision) else math.ceil(precision)
            if precision < min_precision:
                precision = min_precision
        mantissa = '%.*f' % (precision, value)
        if not sci_notation:
            return mantissa
        s = '%s\\cdot10^{%d}' % (mantissa, exponent)
        return s

    def _one_minus(self, s):
        if self._use_overline:
            return '\\overline{%s}' % s
        else:
            return f'1-{s}'

    def __call__(self, x, pos=None):
        if self._minor and x not in self._labelled:
            return ''
        if x <= 0 or x >= 1:
            return ''
        if _is_close_to_int(2 * x) and round(2 * x) == 1:
            s = self._one_half
        elif x < 0.5 and _is_decade(x, rtol=1e-07):
            exponent = round(math.log10(x))
            s = '10^{%d}' % exponent
        elif x > 0.5 and _is_decade(1 - x, rtol=1e-07):
            exponent = round(math.log10(1 - x))
            s = self._one_minus('10^{%d}' % exponent)
        elif x < 0.1:
            s = self._format_value(x, self.locs)
        elif x > 0.9:
            s = self._one_minus(self._format_value(1 - x, 1 - self.locs))
        else:
            s = self._format_value(x, self.locs, sci_notation=False)
        return '$\\mathdefault{%s}$' % s

    def format_data_short(self, value):
        if value < 0.1:
            return f'{value:e}'
        if value < 0.9:
            return f'{value:f}'
        return f'1-{1 - value:e}'