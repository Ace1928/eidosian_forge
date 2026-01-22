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
def _parse_old(self, s):
    self._check_input(s)
    s = self._apply_rules(s, 'whitespace')
    s = self._replace(s, ' ')
    s = self._apply_rules(s, 'add*_1')
    s = self._apply_rules(s, 'add*_2')
    s = self._convert_function(s)
    s = self._replace(s, '^')
    s = self._apply_rules(s, 'Pi')
    return s