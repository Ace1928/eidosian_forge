import pytest
import pyarrow as pa
import numpy as np
from numba.cuda.cudadrv.devicearray import DeviceNDArray  # noqa: E402
@pytest.mark.parametrize('c', range(len(context_choice_ids)), ids=context_choice_ids)
@pytest.mark.parametrize('dtype', dtypes, ids=dtypes)
def test_numba_memalloc(c, dtype):
    ctx, nb_ctx = context_choices[c]
    dtype = np.dtype(dtype)
    size = 10
    mem = nb_ctx.memalloc(size * dtype.itemsize)
    darr = DeviceNDArray((size,), (dtype.itemsize,), dtype, gpu_data=mem)
    darr[:5] = 99
    darr[5:] = 88
    np.testing.assert_equal(darr.copy_to_host()[:5], 99)
    np.testing.assert_equal(darr.copy_to_host()[5:], 88)
    cbuf = cuda.CudaBuffer.from_numba(mem)
    arr2 = np.frombuffer(cbuf.copy_to_host(), dtype=dtype)
    np.testing.assert_equal(arr2, darr.copy_to_host())