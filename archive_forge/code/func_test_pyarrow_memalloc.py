import pytest
import pyarrow as pa
import numpy as np
from numba.cuda.cudadrv.devicearray import DeviceNDArray  # noqa: E402
@pytest.mark.parametrize('c', range(len(context_choice_ids)), ids=context_choice_ids)
@pytest.mark.parametrize('dtype', dtypes, ids=dtypes)
def test_pyarrow_memalloc(c, dtype):
    ctx, nb_ctx = context_choices[c]
    size = 10
    arr, cbuf = make_random_buffer(size, target='device', dtype=dtype, ctx=ctx)
    mem = cbuf.to_numba()
    darr = DeviceNDArray(arr.shape, arr.strides, arr.dtype, gpu_data=mem)
    np.testing.assert_equal(darr.copy_to_host(), arr)