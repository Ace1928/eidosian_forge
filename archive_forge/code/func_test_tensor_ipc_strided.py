import os
import sys
import pytest
import warnings
import weakref
import numpy as np
import pyarrow as pa
def test_tensor_ipc_strided(tmpdir):
    data1 = np.random.randn(10, 4)
    tensor1 = pa.Tensor.from_numpy(data1[::2])
    data2 = np.random.randn(10, 6, 4)
    tensor2 = pa.Tensor.from_numpy(data2[:, ::2, :])
    path = os.path.join(str(tmpdir), 'pyarrow-tensor-ipc-strided')
    mmap = pa.create_memory_map(path, 2048)
    for tensor in [tensor1, tensor2]:
        mmap.seek(0)
        pa.ipc.write_tensor(tensor, mmap)
        mmap.seek(0)
        result = pa.ipc.read_tensor(mmap)
        assert result.equals(tensor)