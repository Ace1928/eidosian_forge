from __future__ import annotations
import os
import sys
from array import array
import pytest
from dask.multiprocessing import get_context
from dask.sizeof import sizeof
from dask.utils import funcname
def test_register_backend_entrypoint(tmp_path):
    (tmp_path / 'impl_sizeof.py').write_bytes(b'def sizeof_plugin(sizeof):\n    print("REG")\n    @sizeof.register_lazy("class_impl")\n    def register_impl():\n        import class_impl\n        @sizeof.register(class_impl.Impl)\n        def sizeof_impl(obj):\n            return obj.size \n')
    (tmp_path / 'class_impl.py').write_bytes(b'class Impl:\n    def __init__(self, size):\n        self.size = size')
    dist_info = tmp_path / 'impl_sizeof-0.0.0.dist-info'
    dist_info.mkdir()
    (dist_info / 'entry_points.txt').write_bytes(b'[dask.sizeof]\nimpl = impl_sizeof:sizeof_plugin\n')
    with get_context().Pool(1) as pool:
        assert pool.apply(_get_sizeof_on_path, args=(tmp_path, 314159265)) == 314159265
    pool.join()