import contextlib
import itertools
import re
import typing
from enum import Enum
from typing import Callable
import sympy
from sympy import Add, Implies, sqrt
from sympy.core import Mul, Pow
from sympy.core import (S, pi, symbols, Function, Rational, Integer,
from sympy.functions import Piecewise, exp, sin, cos
from sympy.printing.smtlib import smtlib_code
from sympy.testing.pytest import raises, Failed
def test_smtlib_boolean():
    with _check_warns([]) as w:
        assert smtlib_code(True, auto_assert=False, log_warn=w) == 'true'
        assert smtlib_code(True, log_warn=w) == '(assert true)'
        assert smtlib_code(S.true, log_warn=w) == '(assert true)'
        assert smtlib_code(S.false, log_warn=w) == '(assert false)'
        assert smtlib_code(False, log_warn=w) == '(assert false)'
        assert smtlib_code(False, auto_assert=False, log_warn=w) == 'false'