from __future__ import annotations
import multiprocessing
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from operator import add
import pytest
import dask
from dask import compute, delayed
from dask.multiprocessing import _dumps, _loads, get, get_context, remote_exception
from dask.system import CPU_COUNT
from dask.utils_test import inc
@pytest.mark.skipif(pickle.HIGHEST_PROTOCOL < 5, reason='requires pickle protocol 5')
def test_out_of_band_pickling():
    """Test that out-of-band pickling works"""
    np = pytest.importorskip('numpy')
    pytest.importorskip('cloudpickle', minversion='1.3.0')
    a = np.arange(5)
    l = []
    b = _dumps(a, buffer_callback=l.append)
    assert len(l) == 1
    assert isinstance(l[0], pickle.PickleBuffer)
    assert memoryview(l[0]) == memoryview(a)
    a2 = _loads(b, buffers=l)
    assert np.all(a == a2)