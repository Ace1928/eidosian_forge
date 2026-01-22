from sympy.core.add import Add
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin, sinc, tan)
from sympy.series.fourier import fourier_series
from sympy.series.fourier import FourierSeries
from sympy.testing.pytest import raises
from functools import lru_cache
@lru_cache()
def _get_examples():
    fo = fourier_series(x, (x, -pi, pi))
    fe = fourier_series(x ** 2, (-pi, pi))
    fp = fourier_series(Piecewise((0, x < 0), (pi, True)), (x, -pi, pi))
    return (fo, fe, fp)