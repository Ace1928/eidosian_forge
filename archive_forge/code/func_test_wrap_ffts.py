from __future__ import annotations
import contextlib
from itertools import combinations_with_replacement
import numpy as np
import pytest
import dask.array as da
import dask.array.fft
from dask.array.core import normalize_chunks
from dask.array.fft import fft_wrap
from dask.array.numpy_compat import NUMPY_GE_200
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('modname', ['numpy.fft', 'scipy.fftpack'])
@pytest.mark.parametrize('funcname', all_1d_funcnames)
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_wrap_ffts(modname, funcname, dtype):
    fft_mod = pytest.importorskip(modname)
    try:
        func = getattr(fft_mod, funcname)
    except AttributeError:
        pytest.skip(f'`{modname}` missing function `{funcname}`.')
    darrc = darr.astype(dtype)
    darr2c = darr2.astype(dtype)
    nparrc = nparr.astype(dtype)
    if modname == 'scipy.fftpack' and 'rfft' in funcname:
        with pytest.raises(ValueError):
            fft_wrap(func)
    else:
        wfunc = fft_wrap(func)
        assert wfunc(darrc).dtype == func(nparrc).dtype
        assert wfunc(darrc).shape == func(nparrc).shape
        assert_eq(wfunc(darrc), func(nparrc))
        assert_eq(wfunc(darrc, axis=1), func(nparrc, axis=1))
        assert_eq(wfunc(darr2c, axis=0), func(nparrc, axis=0))
        assert_eq(wfunc(darrc, n=len(darrc) - 1), func(nparrc, n=len(darrc) - 1))
        assert_eq(wfunc(darrc, axis=1, n=darrc.shape[1] - 1), func(nparrc, n=darrc.shape[1] - 1))
        assert_eq(wfunc(darr2c, axis=0, n=darr2c.shape[0] - 1), func(nparrc, axis=0, n=darr2c.shape[0] - 1))