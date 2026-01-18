from __future__ import annotations
import pytest
import dask.array as da
from dask.array.utils import assert_eq
def test_sparse_hstack_vstack_csr():
    pytest.importorskip('cupyx')
    x = cupy.arange(24, dtype=cupy.float32).reshape(4, 6)
    sp = da.from_array(x, chunks=(2, 3), asarray=False, fancy=False)
    sp = sp.map_blocks(cupyx.scipy.sparse.csr_matrix, dtype=cupy.float32)
    y = sp.compute()
    assert cupyx.scipy.sparse.isspmatrix(y)
    assert_eq(x, y.todense())