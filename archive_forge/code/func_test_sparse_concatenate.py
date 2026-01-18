from __future__ import annotations
import pytest
import dask.array as da
from dask.array.utils import assert_eq
@pytest.mark.parametrize('axis', [0, 1])
def test_sparse_concatenate(axis):
    pytest.importorskip('cupyx')
    rng = da.random.default_rng(cupy.random.default_rng())
    meta = cupyx.scipy.sparse.csr_matrix((0, 0))
    xs = []
    ys = []
    for _ in range(2):
        x = rng.random((1000, 10), chunks=(100, 10))
        x[x < 0.9] = 0
        xs.append(x)
        ys.append(x.map_blocks(cupyx.scipy.sparse.csr_matrix, meta=meta))
    z = da.concatenate(ys, axis=axis)
    z = z.compute()
    if axis == 0:
        sp_concatenate = cupyx.scipy.sparse.vstack
    elif axis == 1:
        sp_concatenate = cupyx.scipy.sparse.hstack
    z_expected = sp_concatenate([cupyx.scipy.sparse.csr_matrix(e.compute()) for e in xs])
    assert (z.toarray() == z_expected.toarray()).all()