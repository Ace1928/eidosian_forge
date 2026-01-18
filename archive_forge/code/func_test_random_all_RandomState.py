from __future__ import annotations
import pytest
import numpy as np
import dask.array as da
from dask import config
from dask.array.utils import assert_eq
@pytest.mark.parametrize('backend', ['cupy', 'numpy'])
@pytest.mark.parametrize('rs', [None, cupy.random.RandomState, np.random.RandomState])
def test_random_all_RandomState(backend, rs):
    if rs == cupy.random.RandomState:
        expect = cupy.ndarray
    elif rs == np.random.RandomState:
        expect = np.ndarray
    elif backend == 'cupy':
        expect = cupy.ndarray
    else:
        expect = np.ndarray

    def rnd_test(func, *args, **kwargs):
        a = func(*args, **kwargs)
        assert type(a._meta) == expect
        assert_eq(a, a)
    with config.set({'array.backend': backend}):
        rs = da.random.RandomState(RandomState=rs)
        rnd_test(rs.beta, 1, 2, size=5, chunks=3)
        rnd_test(rs.binomial, 10, 0.5, size=5, chunks=3)
        rnd_test(rs.chisquare, 1, size=5, chunks=3)
        rnd_test(rs.exponential, 1, size=5, chunks=3)
        rnd_test(rs.f, 1, 2, size=5, chunks=3)
        rnd_test(rs.gamma, 5, 1, size=5, chunks=3)
        rnd_test(rs.geometric, 1, size=5, chunks=3)
        rnd_test(rs.gumbel, 1, size=5, chunks=3)
        rnd_test(rs.hypergeometric, 1, 2, 3, size=5, chunks=3)
        rnd_test(rs.laplace, size=5, chunks=3)
        rnd_test(rs.logistic, size=5, chunks=3)
        rnd_test(rs.lognormal, size=5, chunks=3)
        rnd_test(rs.logseries, 0.5, size=5, chunks=3)
        rnd_test(rs.negative_binomial, 5, 0.5, size=5, chunks=3)
        rnd_test(rs.noncentral_chisquare, 2, 2, size=5, chunks=3)
        rnd_test(rs.noncentral_f, 2, 2, 3, size=5, chunks=3)
        rnd_test(rs.normal, 2, 2, size=5, chunks=3)
        rnd_test(rs.pareto, 1, size=5, chunks=3)
        rnd_test(rs.poisson, size=5, chunks=3)
        rnd_test(rs.power, 1, size=5, chunks=3)
        rnd_test(rs.rayleigh, size=5, chunks=3)
        rnd_test(rs.random_sample, size=5, chunks=3)
        rnd_test(rs.triangular, 1, 2, 3, size=5, chunks=3)
        rnd_test(rs.uniform, size=5, chunks=3)
        rnd_test(rs.vonmises, 2, 3, size=5, chunks=3)
        rnd_test(rs.wald, 1, 2, size=5, chunks=3)
        rnd_test(rs.weibull, 2, size=5, chunks=3)
        rnd_test(rs.zipf, 2, size=5, chunks=3)
        rnd_test(rs.standard_cauchy, size=5, chunks=3)
        rnd_test(rs.standard_exponential, size=5, chunks=3)
        rnd_test(rs.standard_gamma, 2, size=5, chunks=3)
        rnd_test(rs.standard_normal, size=5, chunks=3)
        rnd_test(rs.standard_t, 2, size=5, chunks=3)