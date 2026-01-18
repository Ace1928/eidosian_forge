import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
@pytest.mark.parametrize('method', quantile_methods)
@pytest.mark.parametrize('alpha', [0.2, 0.5, 0.9])
def test_quantile_add_and_multiply_constant(self, method, alpha):
    rng = np.random.default_rng(4321)
    n = 102
    y = rng.random(n)
    q = np.quantile(y, alpha, method=method)
    c = 13.5
    assert_allclose(np.quantile(c + y, alpha, method=method), c + q)
    assert_allclose(np.quantile(c * y, alpha, method=method), c * q)
    q = -np.quantile(-y, 1 - alpha, method=method)
    if method == 'inverted_cdf':
        if n * alpha == int(n * alpha) or np.round(n * alpha) == int(n * alpha) + 1:
            assert_allclose(q, np.quantile(y, alpha, method='higher'))
        else:
            assert_allclose(q, np.quantile(y, alpha, method='lower'))
    elif method == 'closest_observation':
        if n * alpha == int(n * alpha):
            assert_allclose(q, np.quantile(y, alpha, method='higher'))
        elif np.round(n * alpha) == int(n * alpha) + 1:
            assert_allclose(q, np.quantile(y, alpha + 1 / n, method='higher'))
        else:
            assert_allclose(q, np.quantile(y, alpha, method='lower'))
    elif method == 'interpolated_inverted_cdf':
        assert_allclose(q, np.quantile(y, alpha + 1 / n, method=method))
    elif method == 'nearest':
        if n * alpha == int(n * alpha):
            assert_allclose(q, np.quantile(y, alpha + 1 / n, method=method))
        else:
            assert_allclose(q, np.quantile(y, alpha, method=method))
    elif method == 'lower':
        assert_allclose(q, np.quantile(y, alpha, method='higher'))
    elif method == 'higher':
        assert_allclose(q, np.quantile(y, alpha, method='lower'))
    else:
        assert_allclose(q, np.quantile(y, alpha, method=method))