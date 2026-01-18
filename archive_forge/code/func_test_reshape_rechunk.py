from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.reshape import contract_tuple, expand_tuple, reshape_rechunk
from dask.array.utils import assert_eq
@pytest.mark.parametrize('inshape,outshape,prechunks,inchunks,outchunks', [((4,), (4,), ((2, 2),), ((2, 2),), ((2, 2),)), ((4,), (2, 2), ((2, 2),), ((2, 2),), ((1, 1), (2,))), ((4,), (4, 1), ((2, 2),), ((2, 2),), ((2, 2), (1,))), ((4,), (1, 4), ((2, 2),), ((2, 2),), ((1,), (2, 2))), ((1, 4), (4,), ((1,), (2, 2)), ((1,), (2, 2)), ((2, 2),)), ((4, 1), (4,), ((2, 2), (1,)), ((2, 2), (1,)), ((2, 2),)), ((4, 1, 4), (4, 4), ((2, 2), (1,), (2, 2)), ((2, 2), (1,), (2, 2)), ((2, 2), (2, 2))), ((4, 4), (4, 1, 4), ((2, 2), (2, 2)), ((2, 2), (2, 2)), ((2, 2), (1,), (2, 2))), ((2, 2), (4,), ((2,), (2,)), ((2,), (2,)), ((4,),)), ((2, 2), (4,), ((1, 1), (2,)), ((1, 1), (2,)), ((2, 2),)), ((2, 2), (4,), ((2,), (1, 1)), ((1, 1), (2,)), ((2, 2),)), ((64,), (4, 4, 4), ((8, 8, 8, 8, 8, 8, 8, 8),), ((16, 16, 16, 16),), ((1, 1, 1, 1), (4,), (4,))), ((64,), (4, 4, 4), ((32, 32),), ((32, 32),), ((2, 2), (4,), (4,))), ((64,), (4, 4, 4), ((16, 48),), ((16, 48),), ((1, 3), (4,), (4,))), ((64,), (4, 4, 4), ((20, 44),), ((16, 48),), ((1, 3), (4,), (4,))), ((64, 4), (8, 8, 4), ((16, 16, 16, 16), (2, 2)), ((16, 16, 16, 16), (2, 2)), ((2, 2, 2, 2), (8,), (2, 2)))])
def test_reshape_rechunk(inshape, outshape, prechunks, inchunks, outchunks):
    result_in, result_out = reshape_rechunk(inshape, outshape, prechunks)
    assert result_in == inchunks
    assert result_out == outchunks
    assert np.prod(list(map(len, result_in))) == np.prod(list(map(len, result_out)))