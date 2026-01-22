from __future__ import annotations
from typing import Tuple as tTuple, Optional, Union as tUnion, Callable, List, Dict as tDict, Type, TYPE_CHECKING, \
import math
import mpmath.libmp as libmp
from mpmath import (
from mpmath import inf as mpmath_inf
from mpmath.libmp import (from_int, from_man_exp, from_rational, fhalf,
from mpmath.libmp import bitcount as mpmath_bitcount
from mpmath.libmp.backend import MPZ
from mpmath.libmp.libmpc import _infs_nan
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps
from .sympify import sympify
from .singleton import S
from sympy.external.gmpy import SYMPY_INTS
from sympy.utilities.iterables import is_sequence
from sympy.utilities.lambdify import lambdify
from sympy.utilities.misc import as_int
def _create_evalf_table():
    global evalf_table
    from sympy.concrete.products import Product
    from sympy.concrete.summations import Sum
    from .add import Add
    from .mul import Mul
    from .numbers import Exp1, Float, Half, ImaginaryUnit, Integer, NaN, NegativeOne, One, Pi, Rational, Zero, ComplexInfinity, AlgebraicNumber
    from .power import Pow
    from .symbol import Dummy, Symbol
    from sympy.functions.elementary.complexes import Abs, im, re
    from sympy.functions.elementary.exponential import exp, log
    from sympy.functions.elementary.integers import ceiling, floor
    from sympy.functions.elementary.piecewise import Piecewise
    from sympy.functions.elementary.trigonometric import atan, cos, sin
    from sympy.integrals.integrals import Integral
    evalf_table = {Symbol: evalf_symbol, Dummy: evalf_symbol, Float: evalf_float, Rational: evalf_rational, Integer: evalf_integer, Zero: lambda x, prec, options: (None, None, prec, None), One: lambda x, prec, options: (fone, None, prec, None), Half: lambda x, prec, options: (fhalf, None, prec, None), Pi: lambda x, prec, options: (mpf_pi(prec), None, prec, None), Exp1: lambda x, prec, options: (mpf_e(prec), None, prec, None), ImaginaryUnit: lambda x, prec, options: (None, fone, None, prec), NegativeOne: lambda x, prec, options: (fnone, None, prec, None), ComplexInfinity: lambda x, prec, options: S.ComplexInfinity, NaN: lambda x, prec, options: (fnan, None, prec, None), exp: evalf_exp, cos: evalf_trig, sin: evalf_trig, Add: evalf_add, Mul: evalf_mul, Pow: evalf_pow, log: evalf_log, atan: evalf_atan, Abs: evalf_abs, re: evalf_re, im: evalf_im, floor: evalf_floor, ceiling: evalf_ceiling, Integral: evalf_integral, Sum: evalf_sum, Product: evalf_prod, Piecewise: evalf_piecewise, AlgebraicNumber: evalf_alg_num}