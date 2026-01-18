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
def test_square_wave():
    """Test if fourier_series approximates discontinuous function correctly."""
    square_wave = Piecewise((1, x < pi), (-1, True))
    s = fourier_series(square_wave, (x, 0, 2 * pi))
    assert s.truncate(3) == 4 / pi * sin(x) + 4 / (3 * pi) * sin(3 * x) + 4 / (5 * pi) * sin(5 * x)
    assert s.sigma_approximation(4) == 4 / pi * sin(x) * sinc(pi / 4) + 4 / (3 * pi) * sin(3 * x) * sinc(3 * pi / 4)