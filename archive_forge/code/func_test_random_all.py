from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq
from dask.multiprocessing import _dumps, _loads
from dask.utils import key_split
@pytest.mark.parametrize('sz', [None, 5, (2, 2)], ids=type)
def test_random_all(sz):
    da.random.beta(1, 2, size=sz, chunks=3).compute()
    da.random.binomial(10, 0.5, size=sz, chunks=3).compute()
    da.random.chisquare(1, size=sz, chunks=3).compute()
    da.random.exponential(1, size=sz, chunks=3).compute()
    da.random.f(1, 2, size=sz, chunks=3).compute()
    da.random.gamma(5, 1, size=sz, chunks=3).compute()
    da.random.geometric(1, size=sz, chunks=3).compute()
    da.random.gumbel(1, size=sz, chunks=3).compute()
    da.random.hypergeometric(1, 2, 3, size=sz, chunks=3).compute()
    da.random.laplace(size=sz, chunks=3).compute()
    da.random.logistic(size=sz, chunks=3).compute()
    da.random.lognormal(size=sz, chunks=3).compute()
    da.random.logseries(0.5, size=sz, chunks=3).compute()
    da.random.multinomial(20, [1 / 6.0] * 6, size=sz, chunks=3).compute()
    da.random.negative_binomial(5, 0.5, size=sz, chunks=3).compute()
    da.random.noncentral_chisquare(2, 2, size=sz, chunks=3).compute()
    da.random.noncentral_f(2, 2, 3, size=sz, chunks=3).compute()
    da.random.normal(2, 2, size=sz, chunks=3).compute()
    da.random.pareto(1, size=sz, chunks=3).compute()
    da.random.poisson(size=sz, chunks=3).compute()
    da.random.power(1, size=sz, chunks=3).compute()
    da.random.rayleigh(size=sz, chunks=3).compute()
    da.random.triangular(1, 2, 3, size=sz, chunks=3).compute()
    da.random.uniform(size=sz, chunks=3).compute()
    da.random.vonmises(2, 3, size=sz, chunks=3).compute()
    da.random.wald(1, 2, size=sz, chunks=3).compute()
    da.random.weibull(2, size=sz, chunks=3).compute()
    da.random.zipf(2, size=sz, chunks=3).compute()
    da.random.standard_cauchy(size=sz, chunks=3).compute()
    da.random.standard_exponential(size=sz, chunks=3).compute()
    da.random.standard_gamma(2, size=sz, chunks=3).compute()
    da.random.standard_normal(size=sz, chunks=3).compute()
    da.random.standard_t(2, size=sz, chunks=3).compute()