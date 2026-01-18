import os
import sys
import pytest
import warnings
import weakref
import numpy as np
import pyarrow as pa
def test_read_tensor(tmpdir):
    data = np.random.randn(10, 4)
    tensor = pa.Tensor.from_numpy(data)
    data_size = pa.ipc.get_tensor_size(tensor)
    path = os.path.join(str(tmpdir), 'pyarrow-tensor-ipc-read-tensor')
    write_mmap = pa.create_memory_map(path, data_size)
    pa.ipc.write_tensor(tensor, write_mmap)
    read_mmap = pa.memory_map(path, mode='r')
    array = pa.ipc.read_tensor(read_mmap).to_numpy()
    np.testing.assert_equal(data, array)