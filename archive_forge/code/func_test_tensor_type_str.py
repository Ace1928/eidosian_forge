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
@pytest.mark.parametrize(('tensor_type', 'text'), [(pa.fixed_shape_tensor(pa.int8(), [2, 2, 3]), 'fixed_shape_tensor[value_type=int8, shape=[2,2,3]]'), (pa.fixed_shape_tensor(pa.int32(), [2, 2, 3], permutation=[0, 2, 1]), 'fixed_shape_tensor[value_type=int32, shape=[2,2,3], permutation=[0,2,1]]'), (pa.fixed_shape_tensor(pa.int64(), [2, 2, 3], dim_names=['C', 'H', 'W']), 'fixed_shape_tensor[value_type=int64, shape=[2,2,3], dim_names=[C,H,W]]')])
def test_tensor_type_str(tensor_type, text):
    tensor_type_str = tensor_type.__str__()
    assert text in tensor_type_str