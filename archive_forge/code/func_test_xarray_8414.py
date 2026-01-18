from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_xarray_8414():
    np = pytest.importorskip('numpy')
    xr = pytest.importorskip('xarray')
    ds = xr.Dataset(data_vars=dict(a=(('x', 'y'), np.arange(80).reshape(8, 10)), b=('y', np.arange(10)))).chunk(x=1)

    def f(ds):
        d = ds.a * ds.b
        return (d * ds.b).sum('y')
    graph = ds.map_blocks(f).__dask_graph__()
    assert (p := max(diagnostics(graph)[1])) == 3, p
    dsk = dict(graph)
    dependencies, dependents = get_deps(dsk)
    roots = {k for k, v in dependencies.items() if not v}
    shared_roots = {r for r in roots if len(dependents[r]) > 1}
    assert len(shared_roots) == 1
    shared_root = shared_roots.pop()
    o = order(dsk)
    assert_topological_sort(dsk, o)
    previous = None
    step = 0
    assert len(dependents[shared_root]) > 2
    for dep in sorted(dependents[shared_root], key=o.__getitem__):
        if previous is not None:
            if step == 0:
                step = o[dep] - o[shared_root] - 1
                assert step > 1
            else:
                assert o[dep] == previous + step
        previous = o[dep]