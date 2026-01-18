from __future__ import annotations
import pytest
import dask
from dask.context import globalmethod
def test_with_get():
    da = pytest.importorskip('dask.array')
    var = [0]

    def myget(dsk, keys, **kwargs):
        var[0] = var[0] + 1
        return dask.get(dsk, keys, **kwargs)
    x = da.ones(10, chunks=(5,))
    assert x.sum().compute() == 10
    assert var[0] == 0
    with dask.config.set(scheduler=myget):
        assert x.sum().compute() == 10
    assert var[0] == 1
    assert x.sum().compute() == 10
    assert var[0] == 1