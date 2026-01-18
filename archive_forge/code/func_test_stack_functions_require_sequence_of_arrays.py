from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.tests.test_dispatch import EncapsulateNDArray, WrappedArray
from dask.array.utils import assert_eq
@pytest.mark.parametrize('func', [lambda x: np.dstack(x), lambda x: np.hstack(x), lambda x: np.vstack(x)])
def test_stack_functions_require_sequence_of_arrays(func):
    x = np.random.default_rng().random((100, 100))
    y = da.from_array(x, chunks=(50, 50))
    with pytest.raises(NotImplementedError, match='expects a sequence of arrays as the first argument'):
        func(y)