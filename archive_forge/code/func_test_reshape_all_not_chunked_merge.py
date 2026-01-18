from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.reshape import contract_tuple, expand_tuple, reshape_rechunk
from dask.array.utils import assert_eq
@pytest.mark.parametrize('inshape, inchunks, expected_inchunks, outshape, outchunks', [((2, 3, 4), ((1, 1), (1, 2), (2, 2)), ((1, 1), (3,), (4,)), (24,), ((12, 12),))])
def test_reshape_all_not_chunked_merge(inshape, inchunks, expected_inchunks, outshape, outchunks):
    base = np.arange(np.prod(inshape)).reshape(inshape)
    a = da.from_array(base, chunks=inchunks)
    inchunks2, outchunks2 = reshape_rechunk(a.shape, outshape, inchunks)
    assert inchunks2 == expected_inchunks
    assert outchunks2 == outchunks
    result = a.reshape(outshape)
    assert result.chunks == outchunks
    assert_eq(result, base.reshape(outshape))