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
def test_random_all_with_class_methods(generator_class, sz):

    def rnd_test(func, *args, **kwargs):
        a = func(*args, **kwargs)
        assert type(a._meta) == np.ndarray
        assert_eq(a, a)
    rnd_test(generator_class().beta, 1, 2, size=sz, chunks=3)
    rnd_test(generator_class().binomial, 10, 0.5, size=sz, chunks=3)
    rnd_test(generator_class().chisquare, 1, size=sz, chunks=3)
    rnd_test(generator_class().exponential, 1, size=sz, chunks=3)
    rnd_test(generator_class().f, 1, 2, size=sz, chunks=3)
    rnd_test(generator_class().gamma, 5, 1, size=sz, chunks=3)
    rnd_test(generator_class().geometric, 1, size=sz, chunks=3)
    rnd_test(generator_class().gumbel, 1, size=sz, chunks=3)
    rnd_test(generator_class().hypergeometric, 1, 2, 3, size=sz, chunks=3)
    rnd_test(generator_class().laplace, size=sz, chunks=3)
    rnd_test(generator_class().logistic, size=sz, chunks=3)
    rnd_test(generator_class().lognormal, size=sz, chunks=3)
    rnd_test(generator_class().logseries, 0.5, size=sz, chunks=3)
    rnd_test(generator_class().multinomial, 20, [1 / 6.0] * 6, size=sz, chunks=3)
    rnd_test(generator_class().negative_binomial, 5, 0.5, size=sz, chunks=3)
    rnd_test(generator_class().noncentral_chisquare, 2, 2, size=sz, chunks=3)
    rnd_test(generator_class().noncentral_f, 2, 2, 3, size=sz, chunks=3)
    rnd_test(generator_class().normal, 2, 2, size=sz, chunks=3)
    rnd_test(generator_class().pareto, 1, size=sz, chunks=3)
    rnd_test(generator_class().poisson, size=sz, chunks=3)
    rnd_test(generator_class().power, 1, size=sz, chunks=3)
    rnd_test(generator_class().rayleigh, size=sz, chunks=3)
    rnd_test(generator_class().triangular, 1, 2, 3, size=sz, chunks=3)
    rnd_test(generator_class().uniform, size=sz, chunks=3)
    rnd_test(generator_class().vonmises, 2, 3, size=sz, chunks=3)
    rnd_test(generator_class().wald, 1, 2, size=sz, chunks=3)
    rnd_test(generator_class().weibull, 2, size=sz, chunks=3)
    rnd_test(generator_class().zipf, 2, size=sz, chunks=3)
    rnd_test(generator_class().standard_cauchy, size=sz, chunks=3)
    rnd_test(generator_class().standard_exponential, size=sz, chunks=3)
    rnd_test(generator_class().standard_gamma, 2, size=sz, chunks=3)
    rnd_test(generator_class().standard_normal, size=sz, chunks=3)
    rnd_test(generator_class().standard_t, 2, size=sz, chunks=3)