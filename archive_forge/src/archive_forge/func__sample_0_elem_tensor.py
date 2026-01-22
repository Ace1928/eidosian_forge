import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
@property
def _sample_0_elem_tensor(self) -> TensorProto:
    np_array = np.random.randn(0, 3).astype(np.float32)
    return helper.make_tensor(name='test', data_type=TensorProto.FLOAT, dims=(0, 3), vals=np_array.reshape(0).tolist())