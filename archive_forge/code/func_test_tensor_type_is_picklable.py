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
def test_tensor_type_is_picklable(pickle_module):
    expected_type = pa.fixed_shape_tensor(pa.int32(), (2, 2))
    result = pickle_module.loads(pickle_module.dumps(expected_type))
    assert result == expected_type
    arr = [[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]]
    storage = pa.array(arr, pa.list_(pa.int32(), 4))
    expected_arr = pa.ExtensionArray.from_storage(expected_type, storage)
    result = pickle_module.loads(pickle_module.dumps(expected_arr))
    assert result == expected_arr