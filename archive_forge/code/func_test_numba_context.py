import pytest
import pyarrow as pa
import numpy as np
from numba.cuda.cudadrv.devicearray import DeviceNDArray  # noqa: E402
@pytest.mark.parametrize('c', range(len(context_choice_ids)), ids=context_choice_ids)
@pytest.mark.parametrize('dtype', dtypes, ids=dtypes)
def test_numba_context(c, dtype):
    ctx, nb_ctx = context_choices[c]
    size = 10
    with nb_cuda.gpus[0]:
        arr, cbuf = make_random_buffer(size, target='device', dtype=dtype, ctx=ctx)
        assert cbuf.context.handle == nb_ctx.handle.value
        mem = cbuf.to_numba()
        darr = DeviceNDArray(arr.shape, arr.strides, arr.dtype, gpu_data=mem)
        np.testing.assert_equal(darr.copy_to_host(), arr)
        darr[0] = 99
        cbuf.context.synchronize()
        arr2 = np.frombuffer(cbuf.copy_to_host(), dtype=dtype)
        assert arr2[0] == 99