from __future__ import annotations
import pytest
import numpy as np
import dask.array as da
from dask.array.utils import assert_eq, same_keys
@pytest.mark.skip
def test_percentile_with_categoricals():
    try:
        import pandas as pd
    except ImportError:
        return
    x0 = pd.Categorical(['Alice', 'Bob', 'Charlie', 'Dennis', 'Alice', 'Alice'])
    x1 = pd.Categorical(['Alice', 'Bob', 'Charlie', 'Dennis', 'Alice', 'Alice'])
    dsk = {('x', 0): x0, ('x', 1): x1}
    x = da.Array(dsk, 'x', chunks=((6, 6),))
    p = da.percentile(x, [50])
    assert (p.compute().categories == x0.categories).all()
    assert (p.compute().codes == [0]).all()
    assert same_keys(da.percentile(x, [50]), da.percentile(x, [50]))