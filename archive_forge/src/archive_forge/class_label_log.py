from __future__ import annotations
import re
import typing
from bisect import bisect_right
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import numpy as np
from .breaks import timedelta_helper
from .utils import (
@dataclass
class label_log:
    """
    Log number labels

    Parameters
    ----------
    base : int
        Base of the logarithm. Default is 10.
    exponent_limits : tuple
        limits (int, int) where if the any of the powers of the
        numbers falls outside, then the labels will be in
        exponent form. This only applies for base 10.
    mathtex : bool
        If True, return the labels in mathtex format as understood
        by Matplotlib.

    Examples
    --------
    >>> label_log()([0.001, 0.1, 100])
    ['0.001', '0.1', '100']

    >>> label_log()([0.0001, 0.1, 10000])
    ['1e-4', '1e-1', '1e4']

    >>> label_log(mathtex=True)([0.0001, 0.1, 10000])
    ['$10^{-4}$', '$10^{-1}$', '$10^{4}$']
    """
    base: float = 10
    exponent_limits: TupleInt2 = (-4, 4)
    mathtex: bool = False

    def _tidyup_labels(self, labels: Sequence[str]) -> Sequence[str]:
        """
        Make all labels uniform in format

        Remove redundant zeros for labels in exponential format.

        Parameters
        ----------
        labels : list-like
            Labels to be tidied.

        Returns
        -------
        out : list-like
            Labels
        """

        def remove_zeroes(s: str) -> str:
            """
            Remove unnecessary zeros for float string s
            """
            tup = s.split('e')
            if len(tup) == 2:
                mantissa = tup[0].rstrip('0').rstrip('.')
                exponent = int(tup[1])
                s = f'{mantissa}e{exponent}' if exponent else mantissa
            return s

        def as_exp(s: str) -> str:
            """
            Float string s as in exponential format
            """
            return s if 'e' in s else '{:1.0e}'.format(float(s))

        def as_mathtex(s: str) -> str:
            """
            Mathtex for maplotlib
            """
            if 'e' not in s:
                assert s == '1', f"Unexpected value s = {s!r}, instead of '1'"
                return f'${self.base}^{{0}}$'
            exp = s.split('e')[1]
            return f'${self.base}^{{{exp}}}$'
        has_e = ['e' in x for x in labels]
        if not all(has_e) and sum(has_e):
            labels = [as_exp(x) for x in labels]
        labels = [remove_zeroes(x) for x in labels]
        has_e = ['e' in x for x in labels]
        if self.mathtex and any(has_e):
            labels = [as_mathtex(x) for x in labels]
        return labels

    def __call__(self, x: FloatArrayLike) -> Sequence[str]:
        """
        Format a sequence of inputs

        Parameters
        ----------
        x : array
            Input

        Returns
        -------
        out : list
            List of strings.
        """
        if len(x) == 0:
            return []
        if self.base == 10:
            xmin = int(np.floor(np.log10(np.min(x))))
            xmax = int(np.ceil(np.log10(np.max(x))))
            emin, emax = self.exponent_limits
            all_multiples = np.all([np.log10(num).is_integer() for num in x])
            beyond_threshold = xmin <= emin or emax <= xmax
            use_exponents = (same_log10_order_of_magnitude(x) or all_multiples) and beyond_threshold
            fmt = '{:1.0e}' if use_exponents else '{:g}'
            labels = [fmt.format(num) for num in x]
            return self._tidyup_labels(labels)
        else:

            def _exp(num, base):
                e = np.log(num) / np.log(base)
                e_round = np.round(e)
                e = int(e_round) if np.isclose(e, e_round) else np.round(e, 3)
                return e
            base_txt = f'{self.base}'
            if self.base == np.e:
                base_txt = 'e'
            if self.mathtex:
                fmt_parts = (f'${base_txt}^', '{{{e}}}$')
            else:
                fmt_parts = (f'{base_txt}^', '{e}')
            fmt = ''.join(fmt_parts)
            exps = [_exp(num, self.base) for num in x]
            labels = [fmt.format(e=e) for e in exps]
            return labels