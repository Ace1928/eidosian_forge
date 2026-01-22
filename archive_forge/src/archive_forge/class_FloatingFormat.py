import functools
import numbers
import sys
import numpy as np
from . import numerictypes as _nt
from .umath import absolute, isinf, isfinite, isnat
from . import multiarray
from .multiarray import (array, dragon4_positional, dragon4_scientific,
from .fromnumeric import any
from .numeric import concatenate, asarray, errstate
from .numerictypes import (longlong, intc, int_, float_, complex_, bool_,
from .overrides import array_function_dispatch, set_module
import operator
import warnings
import contextlib
class FloatingFormat:
    """ Formatter for subtypes of np.floating """

    def __init__(self, data, precision, floatmode, suppress_small, sign=False, *, legacy=None):
        if isinstance(sign, bool):
            sign = '+' if sign else '-'
        self._legacy = legacy
        if self._legacy <= 113:
            if data.shape != () and sign == '-':
                sign = ' '
        self.floatmode = floatmode
        if floatmode == 'unique':
            self.precision = None
        else:
            self.precision = precision
        self.precision = _none_or_positive_arg(self.precision, 'precision')
        self.suppress_small = suppress_small
        self.sign = sign
        self.exp_format = False
        self.large_exponent = False
        self.fillFormat(data)

    def fillFormat(self, data):
        finite_vals = data[isfinite(data)]
        abs_non_zero = absolute(finite_vals[finite_vals != 0])
        if len(abs_non_zero) != 0:
            max_val = np.max(abs_non_zero)
            min_val = np.min(abs_non_zero)
            with errstate(over='ignore'):
                if max_val >= 100000000.0 or (not self.suppress_small and (min_val < 0.0001 or max_val / min_val > 1000.0)):
                    self.exp_format = True
        if len(finite_vals) == 0:
            self.pad_left = 0
            self.pad_right = 0
            self.trim = '.'
            self.exp_size = -1
            self.unique = True
            self.min_digits = None
        elif self.exp_format:
            trim, unique = ('.', True)
            if self.floatmode == 'fixed' or self._legacy <= 113:
                trim, unique = ('k', False)
            strs = (dragon4_scientific(x, precision=self.precision, unique=unique, trim=trim, sign=self.sign == '+') for x in finite_vals)
            frac_strs, _, exp_strs = zip(*(s.partition('e') for s in strs))
            int_part, frac_part = zip(*(s.split('.') for s in frac_strs))
            self.exp_size = max((len(s) for s in exp_strs)) - 1
            self.trim = 'k'
            self.precision = max((len(s) for s in frac_part))
            self.min_digits = self.precision
            self.unique = unique
            if self._legacy <= 113:
                self.pad_left = 3
            else:
                self.pad_left = max((len(s) for s in int_part))
            self.pad_right = self.exp_size + 2 + self.precision
        else:
            trim, unique = ('.', True)
            if self.floatmode == 'fixed':
                trim, unique = ('k', False)
            strs = (dragon4_positional(x, precision=self.precision, fractional=True, unique=unique, trim=trim, sign=self.sign == '+') for x in finite_vals)
            int_part, frac_part = zip(*(s.split('.') for s in strs))
            if self._legacy <= 113:
                self.pad_left = 1 + max((len(s.lstrip('-+')) for s in int_part))
            else:
                self.pad_left = max((len(s) for s in int_part))
            self.pad_right = max((len(s) for s in frac_part))
            self.exp_size = -1
            self.unique = unique
            if self.floatmode in ['fixed', 'maxprec_equal']:
                self.precision = self.min_digits = self.pad_right
                self.trim = 'k'
            else:
                self.trim = '.'
                self.min_digits = 0
        if self._legacy > 113:
            if self.sign == ' ' and (not any(np.signbit(finite_vals))):
                self.pad_left += 1
        if data.size != finite_vals.size:
            neginf = self.sign != '-' or any(data[isinf(data)] < 0)
            nanlen = len(_format_options['nanstr'])
            inflen = len(_format_options['infstr']) + neginf
            offset = self.pad_right + 1
            self.pad_left = max(self.pad_left, nanlen - offset, inflen - offset)

    def __call__(self, x):
        if not np.isfinite(x):
            with errstate(invalid='ignore'):
                if np.isnan(x):
                    sign = '+' if self.sign == '+' else ''
                    ret = sign + _format_options['nanstr']
                else:
                    sign = '-' if x < 0 else '+' if self.sign == '+' else ''
                    ret = sign + _format_options['infstr']
                return ' ' * (self.pad_left + self.pad_right + 1 - len(ret)) + ret
        if self.exp_format:
            return dragon4_scientific(x, precision=self.precision, min_digits=self.min_digits, unique=self.unique, trim=self.trim, sign=self.sign == '+', pad_left=self.pad_left, exp_digits=self.exp_size)
        else:
            return dragon4_positional(x, precision=self.precision, min_digits=self.min_digits, unique=self.unique, fractional=True, trim=self.trim, sign=self.sign == '+', pad_left=self.pad_left, pad_right=self.pad_right)