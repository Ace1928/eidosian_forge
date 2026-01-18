from __future__ import annotations
import pytest
import numpy as np
import dask.array as da
from dask import config
from dask.array.utils import assert_eq
@pytest.mark.parametrize('backend', ['cupy', 'numpy'])
@pytest.mark.parametrize('gen', [None, cupy.random.default_rng, np.random.default_rng])
@pytest.mark.parametrize('shape', [2, (2, 3), (2, 3, 4), (2, 3, 4, 2)], ids=type)
def test_random_all_Generator(backend, gen, shape):
    if gen == cupy.random.default_rng:
        expect = cupy.ndarray
    elif gen == np.random.default_rng:
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
        generator = gen(5) if gen else None
        rng = da.random.default_rng(generator)
        rnd_test(rng.beta, 1, 2, size=shape, chunks=3)
        rnd_test(rng.binomial, 10, 0.5, size=shape, chunks=3)
        rnd_test(rng.chisquare, 1, size=shape, chunks=3)
        rnd_test(rng.exponential, 1, size=shape, chunks=3)
        rnd_test(rng.f, 1, 2, size=shape, chunks=3)
        rnd_test(rng.gamma, 5, 1, size=shape, chunks=3)
        rnd_test(rng.geometric, 1, size=shape, chunks=3)
        rnd_test(rng.hypergeometric, 1, 2, 3, size=shape, chunks=3)
        rnd_test(rng.integers, 1, high=10, size=shape, chunks=3)
        rnd_test(rng.logseries, 0.5, size=shape, chunks=3)
        rnd_test(rng.poisson, 1, size=shape, chunks=3)
        rnd_test(rng.power, 1, size=shape, chunks=3)
        rnd_test(rng.random, size=shape, chunks=3)
        rnd_test(rng.standard_exponential, size=shape, chunks=3)
        rnd_test(rng.standard_gamma, 2, size=shape, chunks=3)
        rnd_test(rng.standard_normal, size=shape, chunks=3)