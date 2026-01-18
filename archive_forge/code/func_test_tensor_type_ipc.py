import contextlib
import os
import shutil
import subprocess
import weakref
from uuid import uuid4, UUID
import sys
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
@pytest.mark.parametrize('tensor_type', (pa.fixed_shape_tensor(pa.int8(), [2, 2, 3]), pa.fixed_shape_tensor(pa.int8(), [2, 2, 3], permutation=[0, 2, 1]), pa.fixed_shape_tensor(pa.int8(), [2, 2, 3], dim_names=['C', 'H', 'W'])))
def test_tensor_type_ipc(tensor_type):
    storage = pa.array([[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]], pa.list_(pa.int8(), 12))
    arr = pa.ExtensionArray.from_storage(tensor_type, storage)
    batch = pa.RecordBatch.from_arrays([arr], ['ext'])
    tensor_class = tensor_type.__arrow_ext_class__()
    assert isinstance(arr, tensor_class)
    buf = ipc_write_batch(batch)
    del batch
    batch = ipc_read_batch(buf)
    result = batch.column(0)
    assert isinstance(result, tensor_class)
    assert result.type.extension_name == 'arrow.fixed_shape_tensor'
    assert arr.storage.to_pylist() == [[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]]
    assert isinstance(result.type, pa.FixedShapeTensorType)
    assert result.type.value_type == pa.int8()
    assert result.type.shape == [2, 2, 3]