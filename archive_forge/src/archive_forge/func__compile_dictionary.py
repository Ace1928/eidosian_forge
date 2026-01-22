from __future__ import annotations
import re
import typing
from itertools import product
from typing import Any, Callable
import sympy
from sympy import Mul, Add, Pow, log, exp, sqrt, cos, sin, tan, asin, acos, acot, asec, acsc, sinh, cosh, tanh, asinh, \
from sympy.core.sympify import sympify, _sympify
from sympy.functions.special.bessel import airybiprime
from sympy.functions.special.error_functions import li
from sympy.utilities.exceptions import sympy_deprecation_warning
@classmethod
def _compile_dictionary(cls, dic):
    d = {}
    for fm, fs in dic.items():
        cls._check_input(fm)
        cls._check_input(fs)
        fm = cls._apply_rules(fm, 'whitespace')
        fs = cls._apply_rules(fs, 'whitespace')
        fm = cls._replace(fm, ' ')
        fs = cls._replace(fs, ' ')
        m = cls.FM_PATTERN.search(fm)
        if m is None:
            err = "'{f}' function form is invalid.".format(f=fm)
            raise ValueError(err)
        fm_name = m.group()
        args, end = cls._get_args(m)
        if m.start() != 0 or end != len(fm):
            err = "'{f}' function form is invalid.".format(f=fm)
            raise ValueError(err)
        if args[-1][0] == '*':
            key_arg = '*'
        else:
            key_arg = len(args)
        key = (fm_name, key_arg)
        re_args = [x if x[0] != '*' else '\\' + x for x in args]
        xyz = '(?:(' + '|'.join(re_args) + '))'
        patStr = cls.ARGS_PATTERN_TEMPLATE.format(arguments=xyz)
        pat = re.compile(patStr, re.VERBOSE)
        d[key] = {}
        d[key]['fs'] = fs
        d[key]['args'] = args
        d[key]['pat'] = pat
    return d